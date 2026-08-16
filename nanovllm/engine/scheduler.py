from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.outputs import ScheduledRequest, SchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.spec_decode import NgramProposer, verify_greedy_tokens

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_chunk_size = config.prefill_chunk_size
        self.scheduling_policy = config.scheduling_policy
        self.ttft_slo_ms = config.ttft_slo_ms
        self.tpot_slo_ms = config.tpot_slo_ms
        self.max_consecutive_decode_steps = config.max_consecutive_decode_steps
        self.cost_ema_alpha = config.scheduler_cost_ema_alpha
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.speculative_method = getattr(config, "speculative_method", "none")
        self.num_speculative_tokens = getattr(
            config,
            "num_speculative_tokens",
            4,
        )
        self.ngram_proposer = NgramProposer(
            getattr(config, "ngram_min", 2),
            getattr(config, "ngram_max", 5),
        )
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.last_step_was_prefill = False
        self.consecutive_decode_steps = 0
        self.prefill_seconds_per_token: float | None = None
        self.decode_step_seconds: float | None = None
        self.mixed_step_seconds: dict[tuple[int, int], float] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if self.speculative_method != "none" and seq.temperature != 0:
            raise ValueError("speculative decoding requires greedy sampling")
        self.waiting.append(seq)

    def reset_cost_estimates(self):
        self.prefill_seconds_per_token = None
        self.decode_step_seconds = None
        self.mixed_step_seconds.clear()

    def _oldest_ttft_wait_ms(self, now: float | None = None) -> float:
        now = perf_counter() if now is None else now
        ttft_waits = [
            (now - seq.metrics.arrival_time) * 1000
            for seq in self.waiting
            if seq.metrics.first_token_time is None
        ]
        return max(ttft_waits, default=0.0)

    def _should_schedule_prefill(self) -> bool:
        if not self.waiting:
            return False
        if not self.running or self.scheduling_policy == "prefill_first":
            return True
        if self.scheduling_policy == "slo_aware_v2":
            return self._should_schedule_prefill_v2()
        if self.last_step_was_prefill:
            return False
        if self.consecutive_decode_steps >= self.max_consecutive_decode_steps:
            return True
        return self._oldest_ttft_wait_ms() >= self.ttft_slo_ms

    def _remaining_prefill_tokens(self, seq: Sequence) -> int:
        return max(0, seq.num_tokens - seq.num_cached_tokens)

    def _waiting_sequence_slack(
        self,
        seq: Sequence,
        now: float,
    ) -> tuple[float, float]:
        metrics = seq.metrics
        if metrics.first_token_time is None:
            reference = metrics.arrival_time
            target = self.ttft_slo_ms / 1000
        else:
            reference = metrics.last_token_time or metrics.arrival_time
            target = self.tpot_slo_ms / 1000
        predicted = 0.0
        if self.prefill_seconds_per_token is not None:
            predicted = (
                self._remaining_prefill_tokens(seq) * self.prefill_seconds_per_token
            )
        slack = reference + target - now - predicted
        return slack / max(target, 1e-6), slack

    def _waiting_slack(self, now: float) -> tuple[float, float] | None:
        candidates = [self._waiting_sequence_slack(seq, now) for seq in self.waiting]
        return min(candidates, default=None)

    def _promote_most_urgent_waiting(self, now: float):
        if not self.waiting:
            return
        urgent = min(
            self.waiting,
            key=lambda seq: self._waiting_sequence_slack(seq, now)[0],
        )
        if urgent is not self.waiting[0]:
            self.waiting.remove(urgent)
            self.waiting.appendleft(urgent)

    def _decode_slack(self, now: float) -> tuple[float, float] | None:
        target = self.tpot_slo_ms / 1000
        predicted = self.decode_step_seconds or 0.0
        candidates = []
        for seq in self.running:
            metrics = seq.metrics
            reference = (
                metrics.last_token_time
                or metrics.first_token_time
                or metrics.arrival_time
            )
            slack = reference + target - now - predicted
            normalized = slack / max(target, 1e-6)
            candidates.append((normalized, slack))
        return min(candidates, default=None)

    def _sequence_decode_slack(
        self,
        seq: Sequence,
        now: float,
    ) -> tuple[float, float]:
        target = self.tpot_slo_ms / 1000
        reference = (
            seq.metrics.last_token_time
            or seq.metrics.first_token_time
            or seq.metrics.arrival_time
        )
        predicted = self.decode_step_seconds or 0.0
        slack = reference + target - now - predicted
        return slack / max(target, 1e-6), slack

    def _should_schedule_prefill_v2(self) -> bool:
        if self.consecutive_decode_steps >= self.max_consecutive_decode_steps:
            return True
        now = perf_counter()
        waiting_slack = self._waiting_slack(now)
        decode_slack = self._decode_slack(now)
        if waiting_slack is None:
            return False
        if decode_slack is None:
            return True
        return waiting_slack[0] <= decode_slack[0]

    def _prefill_token_budget(self) -> int:
        if self.scheduling_policy == "slo_aware_v2":
            return self._prefill_token_budget_v2()
        if self.scheduling_policy != "slo_aware" or not self.running:
            return self.prefill_chunk_size
        if self._oldest_ttft_wait_ms() >= self.ttft_slo_ms:
            return self.prefill_chunk_size
        return min(self.prefill_chunk_size, self.block_size)

    def _prefill_token_budget_v2(self) -> int:
        if not self.running or self.prefill_seconds_per_token is None:
            return self.prefill_chunk_size
        now = perf_counter()
        waiting_slack = self._waiting_slack(now)
        decode_slack = self._decode_slack(now)
        if waiting_slack is None or waiting_slack[1] <= 0 or decode_slack is None:
            return self.prefill_chunk_size
        safe_tokens = int(max(0.0, decode_slack[1]) / self.prefill_seconds_per_token)
        minimum = min(self.block_size, self.prefill_chunk_size)
        aligned = safe_tokens // self.block_size * self.block_size
        return min(self.prefill_chunk_size, max(minimum, aligned))

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        if self.scheduling_policy == "slo_aware_v2":
            self._promote_most_urgent_waiting(perf_counter())
        token_budget = self._prefill_token_budget()

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = token_budget - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                if seq.metrics.prefill_chunks == 0:
                    seq.metrics.record_prefix_cache_hit(
                        num_cached_blocks * self.block_size
                    )
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # Only the first sequence in a batch may use partial prefill.
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.metrics.mark_scheduled(is_prefill=True)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                seq.metrics.mark_scheduled(is_prefill=False)
                scheduled_seqs.append(seq)
        if self.scheduling_policy in {"slo_aware", "slo_aware_v2"}:
            self.running.extend(scheduled_seqs)
        else:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    @staticmethod
    def _cost_bucket(value: int) -> int:
        if value <= 0:
            return 0
        return 1 << (value - 1).bit_length()

    def _predict_step_seconds(
        self,
        num_prefill_tokens: int,
        num_decode_requests: int,
    ) -> float:
        key = (
            self._cost_bucket(num_prefill_tokens),
            self._cost_bucket(num_decode_requests),
        )
        measured = self.mixed_step_seconds.get(key)
        if measured is not None:
            return measured
        prefill = (
            num_prefill_tokens * self.prefill_seconds_per_token
            if self.prefill_seconds_per_token is not None
            else 0.0
        )
        decode = (
            self.decode_step_seconds
            if num_decode_requests and self.decode_step_seconds is not None
            else 0.0
        )
        return prefill + decode

    def _v3_prefill_chunk_size(
        self,
        seq: Sequence,
        token_budget: int,
        selected_prefill_tokens: int,
        selected_decode_requests: int,
        decode_slacks: list[float],
    ) -> int:
        remaining_tokens = self._remaining_prefill_tokens(seq)
        chunk = min(remaining_tokens, token_budget, self.prefill_chunk_size)
        minimum = min(self.block_size, remaining_tokens, token_budget)
        if chunk <= 0 or chunk == remaining_tokens:
            return chunk
        if self.prefill_seconds_per_token is not None and decode_slacks:
            tightest_slack = max(0.0, min(decode_slacks))
            current_prediction = self._predict_step_seconds(
                selected_prefill_tokens,
                selected_decode_requests,
            )
            available = max(0.0, tightest_slack - current_prediction)
            safe_tokens = int(available / self.prefill_seconds_per_token)
            chunk = min(chunk, max(minimum, safe_tokens))
        if chunk < remaining_tokens:
            aligned = chunk // self.block_size * self.block_size
            chunk = max(minimum, aligned)
        return min(chunk, remaining_tokens, token_budget)

    def _schedule_v3(self) -> SchedulerOutput:
        now = perf_counter()
        candidates: list[tuple[float, float, int, bool, Sequence]] = []
        decode_budget_by_id: dict[int, float] = {}
        for seq in self.running:
            normalized, slack = self._sequence_decode_slack(seq, now)
            decode_budget_by_id[seq.seq_id] = slack + (self.decode_step_seconds or 0.0)
            candidates.append(
                (normalized, seq.metrics.arrival_time, seq.seq_id, False, seq)
            )
        for seq in self.waiting:
            normalized, _ = self._waiting_sequence_slack(seq, now)
            candidates.append(
                (normalized, seq.metrics.arrival_time, seq.seq_id, True, seq)
            )
        candidates.sort(
            key=lambda candidate: (
                candidate[3],
                candidate[0],
                candidate[1],
                candidate[2],
            )
        )

        scheduled: list[ScheduledRequest] = []
        selected_decode: list[Sequence] = []
        selected_ids: set[int] = set()
        token_budget = self.max_num_batched_tokens
        prefill_tokens = 0
        decode_requests = 0

        for _, _, _, is_prefill, seq in candidates:
            if len(scheduled) >= self.max_num_seqs or token_budget <= 0:
                break
            if is_prefill:
                if seq not in self.waiting:
                    continue
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        continue
                    self.block_manager.allocate(seq, num_cached_blocks)
                    if seq.metrics.prefill_chunks == 0:
                        seq.metrics.record_prefix_cache_hit(
                            num_cached_blocks * self.block_size
                        )
                decode_slacks = [
                    budget
                    for seq_id, budget in decode_budget_by_id.items()
                    if seq_id in selected_ids
                ]
                num_tokens = self._v3_prefill_chunk_size(
                    seq,
                    token_budget,
                    prefill_tokens,
                    decode_requests,
                    decode_slacks,
                )
                if num_tokens <= 0:
                    continue
                seq.num_scheduled_tokens = num_tokens
                seq.is_prefill = True
                seq.metrics.mark_scheduled(is_prefill=True)
                needs_sampling = seq.num_cached_tokens + num_tokens == seq.num_tokens
                if needs_sampling:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.remove(seq)
                    self.running.append(seq)
                scheduled.append(
                    ScheduledRequest(
                        sequence=seq,
                        num_scheduled_tokens=num_tokens,
                        is_prefill=True,
                        needs_sampling=needs_sampling,
                    )
                )
                prefill_tokens += num_tokens
                token_budget -= num_tokens
                selected_ids.add(seq.seq_id)
                continue

            if seq not in self.running:
                continue
            max_proposals = min(
                self.num_speculative_tokens,
                max(0, token_budget - 1),
                max(0, seq.max_tokens - seq.num_completion_tokens - 1),
            )
            speculative_token_ids = (
                self.ngram_proposer.propose(seq.token_ids, max_proposals)
                if self.speculative_method == "ngram"
                else []
            )
            num_scheduled_tokens = 1 + len(speculative_token_ids)
            speculative_method = (
                "draft"
                if self.speculative_method == "draft" and max_proposals
                else "ngram"
                if speculative_token_ids
                else "none"
            )
            if speculative_method == "draft":
                num_scheduled_tokens = 1 + max_proposals
            total_tokens = seq.num_cached_tokens + num_scheduled_tokens
            while not self.block_manager.can_reserve(seq, total_tokens):
                victims = [
                    candidate
                    for candidate in self.running
                    if candidate is not seq and candidate.seq_id not in selected_ids
                ]
                if not victims:
                    self.preempt(seq)
                    break
                victim = max(
                    victims,
                    key=lambda candidate: self._sequence_decode_slack(
                        candidate,
                        now,
                    )[0],
                )
                self.preempt(victim)
            else:
                seq.num_scheduled_tokens = num_scheduled_tokens
                seq.is_prefill = False
                self.block_manager.reserve(seq, total_tokens)
                seq.metrics.mark_scheduled(is_prefill=False)
                scheduled.append(
                    ScheduledRequest(
                        sequence=seq,
                        num_scheduled_tokens=num_scheduled_tokens,
                        is_prefill=False,
                        needs_sampling=True,
                        speculative_token_ids=speculative_token_ids,
                        speculative_method=speculative_method,
                    )
                )
                token_budget -= num_scheduled_tokens
                decode_requests += 1
                selected_decode.append(seq)
                selected_ids.add(seq.seq_id)

        for seq in selected_decode:
            if seq in self.running:
                self.running.remove(seq)
                self.running.append(seq)

        if not scheduled:
            sequences = self._schedule_decode()
            if sequences:
                return SchedulerOutput.from_phase(sequences, False)
            sequences = self._schedule_prefill()
            assert sequences
            return SchedulerOutput.from_phase(sequences, True)
        return SchedulerOutput(scheduled)

    def schedule(self) -> SchedulerOutput:
        if self.scheduling_policy == "slo_aware_v3":
            return self._schedule_v3()
        if self._should_schedule_prefill():
            scheduled_seqs = self._schedule_prefill()
            if scheduled_seqs:
                self.last_step_was_prefill = True
                self.consecutive_decode_steps = 0
                return SchedulerOutput.from_phase(scheduled_seqs, True)

        scheduled_seqs = self._schedule_decode()
        if scheduled_seqs:
            self.last_step_was_prefill = False
            self.consecutive_decode_steps += 1
            return SchedulerOutput.from_phase(scheduled_seqs, False)

        scheduled_seqs = self._schedule_prefill()
        assert scheduled_seqs
        self.last_step_was_prefill = True
        self.consecutive_decode_steps = 0
        return SchedulerOutput.from_phase(scheduled_seqs, True)

    def preempt(self, seq: Sequence):
        if seq in self.running:
            self.running.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.metrics.mark_preempted()
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def _update_cost_estimates(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        step_duration: float,
    ):
        if step_duration <= 0:
            return
        if is_prefill:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            if num_tokens == 0:
                return
            sample = step_duration / num_tokens
            current = self.prefill_seconds_per_token
            self.prefill_seconds_per_token = self._ema(current, sample)
        else:
            self.decode_step_seconds = self._ema(
                self.decode_step_seconds,
                step_duration,
            )

    def _update_output_cost_estimates(
        self,
        output: SchedulerOutput,
        step_duration: float,
    ):
        if step_duration <= 0:
            return
        key = (
            self._cost_bucket(output.num_prefill_tokens),
            self._cost_bucket(
                sum(not request.is_prefill for request in output.scheduled_requests)
            ),
        )
        self.mixed_step_seconds[key] = self._ema(
            self.mixed_step_seconds.get(key),
            step_duration,
        )
        if output.is_prefill_only:
            self._update_cost_estimates(
                output.sequences,
                True,
                step_duration,
            )
        elif output.is_decode_only:
            self._update_cost_estimates(
                output.sequences,
                False,
                step_duration,
            )

    def _ema(self, current: float | None, sample: float) -> float:
        if current is None:
            return sample
        alpha = self.cost_ema_alpha
        return alpha * sample + (1 - alpha) * current

    def postprocess(
        self,
        output: SchedulerOutput | list[Sequence],
        token_ids: list[int | list[int] | None],
        is_prefill: bool | None = None,
        step_duration: float = 0.0,
        step_finished_at: float | None = None,
    ):
        step_finished_at = (
            perf_counter() if step_finished_at is None else step_finished_at
        )
        if isinstance(output, SchedulerOutput):
            scheduler_output = output
            self._update_output_cost_estimates(
                scheduler_output,
                step_duration,
            )
        else:
            assert is_prefill is not None
            scheduler_output = SchedulerOutput.from_phase(output, is_prefill)
            self._update_cost_estimates(output, is_prefill, step_duration)
        for request, token_id in zip(
            scheduler_output.scheduled_requests,
            token_ids,
        ):
            seq = request.sequence
            seq.metrics.record_model_step(request.is_prefill, step_duration)
            if request.speculative_token_ids:
                assert isinstance(token_id, list)
                self._postprocess_speculative(
                    request,
                    token_id,
                    step_finished_at,
                )
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += request.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if request.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                seq.metrics.mark_queued(step_finished_at)
                continue
            assert token_id is not None
            seq.append_token(token_id)
            seq.metrics.mark_token(
                seq.num_completion_tokens,
                step_finished_at,
            )
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.metrics.mark_finished(step_finished_at)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def _postprocess_speculative(
        self,
        request: ScheduledRequest,
        target_token_ids: list[int],
        step_finished_at: float,
    ):
        seq = request.sequence
        verification = verify_greedy_tokens(
            request.speculative_token_ids,
            target_token_ids,
        )
        remaining_tokens = seq.max_tokens - seq.num_completion_tokens
        emitted: list[int] = []
        for token_id in verification.output_token_ids[:remaining_tokens]:
            emitted.append(token_id)
            if not seq.ignore_eos and token_id == self.eos:
                break

        request.accepted_tokens = min(
            verification.accepted_tokens,
            len(emitted),
        )
        seq.metrics.record_speculation(
            len(request.speculative_token_ids),
            request.accepted_tokens,
        )
        committed_tokens = 1 + request.accepted_tokens
        seq.num_scheduled_tokens = committed_tokens
        for token_id in emitted:
            seq.append_token(token_id)
            seq.metrics.mark_token(
                seq.num_completion_tokens,
                step_finished_at,
            )
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += committed_tokens
        seq.draft_num_cached_tokens = min(
            seq.draft_num_cached_tokens,
            seq.num_cached_tokens,
        )
        seq.num_scheduled_tokens = 0
        self.block_manager.truncate(seq, len(seq))

        if not emitted:
            raise RuntimeError("speculative verification emitted no token")
        last_token = emitted[-1]
        if (
            not seq.ignore_eos and last_token == self.eos
        ) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            seq.metrics.mark_finished(step_finished_at)
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
