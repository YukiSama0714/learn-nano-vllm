import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from time import perf_counter

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.outputs import ModelRunnerOutput, SchedulerOutput
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.use_cudagraph = (
            not config.enforce_eager and config.attention_backend == "flash_attn"
        )
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(
            hf_config,
            rms_norm_backend=config.rms_norm_backend,
            attention_backend=config.attention_backend,
            block_size=config.kvcache_block_size,
        )
        load_model(self.model, config.model)
        self.draft_model = None
        if config.speculative_method == "draft":
            draft_hf_config = config.draft_hf_config
            torch.set_default_dtype(draft_hf_config.dtype)
            self.draft_model = Qwen3ForCausalLM(
                draft_hf_config,
                rms_norm_backend=config.rms_norm_backend,
                attention_backend=config.attention_backend,
                block_size=config.kvcache_block_size,
            )
            load_model(self.draft_model, config.draft_model)
            torch.set_default_dtype(hf_config.dtype)
        self.sampler = Sampler()
        self.warmup_model()
        if self.draft_model is not None:
            self.warmup_draft_model()
        self.allocate_kv_cache()
        if self.use_cudagraph:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.use_cudagraph:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(SchedulerOutput.from_phase(seqs, True))
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup_draft_model(self):
        seq_len = min(
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        input_ids = torch.zeros(seq_len, dtype=torch.int64)
        positions = torch.arange(seq_len, dtype=torch.int64)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=torch.empty(0, dtype=torch.int32),
            query_to_request=torch.zeros(seq_len, dtype=torch.int32),
            query_positions=positions.to(torch.int32),
        )
        hidden_states = self.draft_model(input_ids, positions)
        self.draft_model.compute_logits(hidden_states[-1:])
        reset_context()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        block_bytes = self._model_kv_block_bytes(hf_config)
        if config.draft_hf_config is not None:
            block_bytes += self._model_kv_block_bytes(config.draft_hf_config)
        available_bytes = int(total * config.gpu_memory_utilization)
        available_bytes -= used + peak - current
        config.num_kvcache_blocks = available_bytes // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = self._allocate_model_kv_cache(hf_config)
        self._bind_kv_cache(self.model, self.kv_cache)
        if self.draft_model is not None:
            self.draft_kv_cache = self._allocate_model_kv_cache(config.draft_hf_config)
            self._bind_kv_cache(self.draft_model, self.draft_kv_cache)

    def _model_kv_geometry(self, hf_config):
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        return num_kv_heads, head_dim

    def _model_kv_block_bytes(self, hf_config) -> int:
        num_kv_heads, head_dim = self._model_kv_geometry(hf_config)
        return (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )

    def _allocate_model_kv_cache(self, hf_config):
        num_kv_heads, head_dim = self._model_kv_geometry(hf_config)
        return torch.empty(
            2,
            hf_config.num_hidden_layers,
            self.config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=hf_config.dtype,
        )

    @staticmethod
    def _bind_kv_cache(model, kv_cache):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_batch(self, output: SchedulerOutput):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        context_lens = []
        slot_mapping = []
        logits_indices = []
        query_to_request = []
        query_positions = []
        sample_requests: list[tuple[int, int]] = []
        temperatures = []
        seqs = output.sequences
        has_block_tables = [bool(seq.block_table) for seq in seqs]
        if any(has_block_tables) and not all(has_block_tables):
            raise RuntimeError("scheduled requests must share KV allocation state")
        block_tables = (
            self.prepare_block_tables(seqs) if all(has_block_tables) else None
        )
        for request_index, request in enumerate(output.scheduled_requests):
            seq = request.sequence
            start = seq.num_cached_tokens
            seqlen_q = request.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            context_lens.append(seqlen_k)
            request_input_ids = (
                [seq.last_token] + request.speculative_token_ids
                if request.speculative_token_ids
                else seq[start:end]
            )
            if len(request_input_ids) != seqlen_q:
                raise RuntimeError("scheduled query length does not match input")
            input_ids.extend(request_input_ids)
            request_positions = range(start, end)
            positions.extend(request_positions)
            query_positions.extend(request_positions)
            query_to_request.extend([request_index] * seqlen_q)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if request.needs_sampling:
                num_logits = 1 + len(request.speculative_token_ids)
                logits_start = cu_seqlens_q[-1] - num_logits
                logits_indices.extend(range(logits_start, cu_seqlens_q[-1]))
                sample_requests.append((request_index, num_logits))
                temperatures.extend([seq.temperature] * num_logits)
            if seq.block_table:
                for token_index in range(start, end):
                    block_index, block_offset = divmod(
                        token_index,
                        self.block_size,
                    )
                    slot_mapping.append(
                        seq.block_table[block_index] * self.block_size + block_offset
                    )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        logits_indices = torch.tensor(
            logits_indices,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)
        temperatures = torch.tensor(
            temperatures,
            dtype=torch.float32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        query_to_request = torch.tensor(
            query_to_request,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        query_positions = torch.tensor(
            query_positions,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        set_context(
            not output.is_decode_only,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens,
            block_tables,
            logits_indices,
            output.is_mixed,
            query_to_request,
            query_positions,
        )
        return (
            input_ids,
            positions,
            temperatures,
            sample_requests,
        )

    @torch.inference_mode()
    def prepare_draft_speculation(self, output: SchedulerOutput):
        for request in output.scheduled_requests:
            if request.speculative_method != "draft":
                continue
            num_proposals = request.num_scheduled_tokens - 1
            request.speculative_token_ids = self._propose_draft_tokens(
                request.sequence,
                num_proposals,
            )

    def _propose_draft_tokens(
        self,
        seq: Sequence,
        num_proposals: int,
    ) -> list[int]:
        assert self.draft_model is not None
        seq.draft_num_cached_tokens = min(
            seq.draft_num_cached_tokens, seq.num_cached_tokens
        )
        start = seq.draft_num_cached_tokens
        proposals = [self._run_draft_query(seq, seq[start:], start)]
        seq.draft_num_cached_tokens = len(seq)
        for _ in range(1, num_proposals):
            start = seq.draft_num_cached_tokens
            proposals.append(self._run_draft_query(seq, [proposals[-1]], start))
            seq.draft_num_cached_tokens += 1
        return proposals

    def _run_draft_query(
        self,
        seq: Sequence,
        input_token_ids: list[int],
        start: int,
    ) -> int:
        query_length = len(input_token_ids)
        end = start + query_length
        input_ids = torch.tensor(
            input_token_ids,
            dtype=torch.int64,
            device="cuda",
        )
        positions = torch.arange(
            start,
            end,
            dtype=torch.int64,
            device="cuda",
        )
        cu_seqlens_q = torch.tensor(
            [0, query_length],
            dtype=torch.int32,
            device="cuda",
        )
        cu_seqlens_k = torch.tensor(
            [0, end],
            dtype=torch.int32,
            device="cuda",
        )
        slot_mapping = []
        for token_index in range(start, end):
            block_index, block_offset = divmod(
                token_index,
                self.block_size,
            )
            slot_mapping.append(
                seq.block_table[block_index] * self.block_size + block_offset
            )
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=query_length,
            max_seqlen_k=end,
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                device="cuda",
            ),
            context_lens=torch.tensor(
                [end],
                dtype=torch.int32,
                device="cuda",
            ),
            block_tables=self.prepare_block_tables([seq]),
            query_to_request=torch.zeros(
                query_length,
                dtype=torch.int32,
                device="cuda",
            ),
            query_positions=positions.to(torch.int32),
        )
        hidden_states = self.draft_model(input_ids, positions)
        logits = self.draft_model.compute_logits(hidden_states[-1:])
        token_id = logits.argmax(dim=-1).item()
        reset_context()
        return token_id

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        use_cudagraph: bool,
    ):
        if not use_cudagraph or not self.use_cudagraph or input_ids.size(0) > 512:
            return self.model(input_ids, positions)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
            return graph_vars["outputs"][:bs]

    def run(
        self,
        output: SchedulerOutput | list[Sequence],
        is_prefill: bool | None = None,
    ) -> ModelRunnerOutput:
        if not isinstance(output, SchedulerOutput):
            assert is_prefill is not None
            output = SchedulerOutput.from_phase(output, is_prefill)
        prepare_started = perf_counter()
        if self.draft_model is not None:
            self.prepare_draft_speculation(output)
        (
            input_ids,
            positions,
            temperatures,
            sample_requests,
        ) = self.prepare_batch(output)
        prepare_finished = perf_counter()
        model_started = perf_counter()
        hidden_states = self.run_model(
            input_ids,
            positions,
            output.is_decode_only and output.num_proposed_tokens == 0,
        )
        context = get_context()
        if sample_requests:
            logits = self.model.compute_logits(
                hidden_states,
                context.logits_indices,
            )
        else:
            logits = None
        model_finished = perf_counter()
        sampling_started = perf_counter()
        diagnostic_rows: list[dict[str, object]] = []
        if self.rank == 0 and logits is not None:
            sampled = (
                logits.argmax(dim=-1)
                if torch.all(temperatures == 0)
                else self.sampler(logits, temperatures)
            )
            sampled_token_ids = sampled.tolist()
            if self.config.record_token_diagnostics:
                top_logits, top_token_ids = torch.topk(
                    logits.float(),
                    k=2,
                    dim=-1,
                )
                diagnostic_rows = [
                    {
                        "top_token_ids": token_ids,
                        "top_logits": values,
                        "margin": values[0] - values[1],
                    }
                    for token_ids, values in zip(
                        top_token_ids.tolist(),
                        top_logits.tolist(),
                    )
                ]
        else:
            sampled_token_ids = []
        sampling_finished = perf_counter()
        token_ids: list[int | list[int] | None] = [None] * len(
            output.scheduled_requests
        )
        token_diagnostics: list[list[dict[str, object]] | None] = [None] * len(
            output.scheduled_requests
        )
        sample_offset = 0
        for request_index, num_logits in sample_requests:
            request_token_ids = sampled_token_ids[
                sample_offset : sample_offset + num_logits
            ]
            token_ids[request_index] = (
                request_token_ids[0] if num_logits == 1 else request_token_ids
            )
            if diagnostic_rows:
                request = output.scheduled_requests[request_index]
                request_diagnostics = diagnostic_rows[
                    sample_offset : sample_offset + num_logits
                ]
                for diagnostic in request_diagnostics:
                    diagnostic.update(
                        {
                            "mixed_step": output.is_mixed,
                            "is_prefill": request.is_prefill,
                            "query_length": request.num_scheduled_tokens,
                            "context_length": (
                                request.sequence.num_cached_tokens
                                + request.num_scheduled_tokens
                            ),
                        }
                    )
                token_diagnostics[request_index] = request_diagnostics
            sample_offset += num_logits
        reset_context()
        return ModelRunnerOutput(
            token_ids=token_ids,
            token_diagnostics=token_diagnostics,
            input_prep_seconds=prepare_finished - prepare_started,
            model_seconds=model_finished - model_started,
            sampling_seconds=sampling_finished - sampling_started,
        )

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
        }
