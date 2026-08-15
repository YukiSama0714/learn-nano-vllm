from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.prefill_chunk_size = config.prefill_chunk_size
        self.scheduling_policy = config.scheduling_policy
        self.ttft_slo_ms = config.ttft_slo_ms
        self.max_consecutive_decode_steps = config.max_consecutive_decode_steps
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.last_step_was_prefill = False
        self.consecutive_decode_steps = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

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
        if self.last_step_was_prefill:
            return False
        if self.consecutive_decode_steps >= self.max_consecutive_decode_steps:
            return True
        return self._oldest_ttft_wait_ms() >= self.ttft_slo_ms

    def _prefill_token_budget(self) -> int:
        if self.scheduling_policy != "slo_aware" or not self.running:
            return self.prefill_chunk_size
        if self._oldest_ttft_wait_ms() >= self.ttft_slo_ms:
            return self.prefill_chunk_size
        return min(self.prefill_chunk_size, self.block_size)

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
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
        if self.scheduling_policy == "slo_aware":
            self.running.extend(scheduled_seqs)
        else:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self._should_schedule_prefill():
            scheduled_seqs = self._schedule_prefill()
            if scheduled_seqs:
                self.last_step_was_prefill = True
                self.consecutive_decode_steps = 0
                return scheduled_seqs, True

        scheduled_seqs = self._schedule_decode()
        if scheduled_seqs:
            self.last_step_was_prefill = False
            self.consecutive_decode_steps += 1
            return scheduled_seqs, False

        scheduled_seqs = self._schedule_prefill()
        assert scheduled_seqs
        self.last_step_was_prefill = True
        self.consecutive_decode_steps = 0
        return scheduled_seqs, True

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.metrics.mark_preempted()
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
        step_duration: float = 0.0,
        step_finished_at: float | None = None,
    ):
        step_finished_at = (
            perf_counter() if step_finished_at is None else step_finished_at
        )
        for seq, token_id in zip(seqs, token_ids):
            seq.metrics.record_model_step(is_prefill, step_duration)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                seq.metrics.mark_queued(step_finished_at)
                continue
            seq.append_token(token_id)
            seq.metrics.mark_token(
                seq.num_completion_tokens,
                step_finished_at,
            )
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                seq.metrics.mark_finished(step_finished_at)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
