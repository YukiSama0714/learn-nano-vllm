import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._finished_request_metrics = {}
        self._step_metrics = []
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        arrival_time: float | None = None,
    ):
        arrival_time = perf_counter() if arrival_time is None else arrival_time
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, arrival_time)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        schedule_started_at = perf_counter()
        scheduler_output = self.scheduler.schedule()
        scheduled_at = perf_counter()
        num_tokens = scheduler_output.num_prefill_tokens
        if num_tokens == 0:
            num_tokens = -scheduler_output.num_decode_tokens
        started_at = perf_counter()
        model_output = self.model_runner.call("run", scheduler_output)
        finished_at = perf_counter()
        if model_output.token_diagnostics:
            for request, diagnostics in zip(
                scheduler_output.scheduled_requests,
                model_output.token_diagnostics,
            ):
                if diagnostics:
                    request.sequence.metrics.record_token_diagnostics(diagnostics)
        self.scheduler.postprocess(
            scheduler_output,
            model_output.token_ids,
            step_duration=finished_at - started_at,
            step_finished_at=finished_at,
        )
        block_stats = self.scheduler.block_manager.stats(
            list(self.scheduler.waiting) + list(self.scheduler.running)
        )
        self._step_metrics.append(
            {
                "scheduler_ms": round(
                    (scheduled_at - schedule_started_at) * 1000,
                    3,
                ),
                "input_prep_ms": round(
                    model_output.input_prep_seconds * 1000,
                    3,
                ),
                "model_ms": round(model_output.model_seconds * 1000, 3),
                "sampling_ms": round(
                    model_output.sampling_seconds * 1000,
                    3,
                ),
                "prefill_tokens": scheduler_output.num_prefill_tokens,
                "decode_tokens": scheduler_output.num_decode_tokens,
                "mixed": scheduler_output.is_mixed,
                "proposed_tokens": scheduler_output.num_proposed_tokens,
                "accepted_tokens": scheduler_output.num_accepted_tokens,
            }
            | block_stats
        )
        for seq in scheduler_output.sequences:
            if seq.is_finished:
                self._finished_request_metrics[seq.seq_id] = seq.metrics.to_dict(
                    seq.num_prompt_tokens,
                    seq.num_completion_tokens,
                )
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in scheduler_output.sequences
            if seq.is_finished
        ]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def take_finished_request_metrics(self, seq_id: int) -> dict:
        return self._finished_request_metrics.pop(seq_id)

    def reset_step_metrics(self):
        self._step_metrics.clear()

    def take_step_metrics(self) -> list[dict]:
        metrics, self._step_metrics = self._step_metrics, []
        return metrics

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("prompts and sampling_params must have equal lengths")
        arrival_time = perf_counter()
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp, arrival_time)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "metrics": self.take_finished_request_metrics(seq_id),
            }
            for seq_id, token_ids in sorted(outputs.items())
        ]
        return outputs
