from dataclasses import dataclass, field
from time import perf_counter


def _milliseconds(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(seconds * 1000, 3)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


@dataclass(slots=True)
class RequestMetrics:
    arrival_time: float = field(default_factory=perf_counter)
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    queued_at: float | None = field(init=False)
    queue_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    prefix_cache_hit_tokens: int = 0
    prefill_chunks: int = 0
    decode_steps: int = 0
    preemptions: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    inter_token_gaps: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.queued_at = self.arrival_time

    def mark_scheduled(self, is_prefill: bool, now: float | None = None):
        now = perf_counter() if now is None else now
        if self.first_scheduled_time is None:
            self.first_scheduled_time = now
        if self.queued_at is not None:
            self.queue_time += now - self.queued_at
            self.queued_at = None
        if is_prefill:
            self.prefill_chunks += 1
        else:
            self.decode_steps += 1

    def mark_queued(self, now: float | None = None):
        if self.queued_at is None:
            self.queued_at = perf_counter() if now is None else now

    def record_model_step(self, is_prefill: bool, duration: float):
        if is_prefill:
            self.prefill_time += duration
        else:
            self.decode_time += duration

    def record_prefix_cache_hit(self, num_tokens: int):
        self.prefix_cache_hit_tokens = max(
            self.prefix_cache_hit_tokens,
            num_tokens,
        )

    def mark_token(self, num_completion_tokens: int, now: float | None = None):
        now = perf_counter() if now is None else now
        if num_completion_tokens == 1 and self.first_token_time is None:
            self.first_token_time = now
        elif self.last_token_time is not None:
            self.inter_token_gaps.append(now - self.last_token_time)
        self.last_token_time = now

    def mark_finished(self, now: float | None = None):
        self.finish_time = perf_counter() if now is None else now

    def mark_preempted(self, now: float | None = None):
        self.preemptions += 1
        self.mark_queued(now)

    def record_speculation(self, proposed: int, accepted: int):
        self.proposed_tokens += proposed
        self.accepted_tokens += accepted

    def current_queue_time(self, now: float | None = None) -> float:
        queue_time = self.queue_time
        if self.queued_at is not None:
            now = perf_counter() if now is None else now
            queue_time += now - self.queued_at
        return queue_time

    def to_dict(
        self,
        num_prompt_tokens: int,
        num_completion_tokens: int,
    ) -> dict[str, float | int | None]:
        ttft = None
        e2e = None
        tpot = None
        if self.first_token_time is not None:
            ttft = self.first_token_time - self.arrival_time
        if self.finish_time is not None:
            e2e = self.finish_time - self.arrival_time
        if (
            self.first_token_time is not None
            and self.finish_time is not None
            and num_completion_tokens > 1
        ):
            tpot = (self.finish_time - self.first_token_time) / (
                num_completion_tokens - 1
            )
        cache_hit_rate = (
            self.prefix_cache_hit_tokens / num_prompt_tokens
            if num_prompt_tokens
            else 0.0
        )
        inter_token_gap_p95 = _percentile(self.inter_token_gaps, 0.95)
        max_inter_token_gap = (
            max(self.inter_token_gaps) if self.inter_token_gaps else None
        )
        return {
            "prompt_tokens": num_prompt_tokens,
            "completion_tokens": num_completion_tokens,
            "ttft_ms": _milliseconds(ttft),
            "tpot_ms": _milliseconds(tpot),
            "e2e_ms": _milliseconds(e2e),
            "queue_ms": _milliseconds(self.queue_time),
            "prefill_ms": _milliseconds(self.prefill_time),
            "decode_ms": _milliseconds(self.decode_time),
            "inter_token_gap_p95_ms": _milliseconds(inter_token_gap_p95),
            "max_inter_token_gap_ms": _milliseconds(max_inter_token_gap),
            "prefix_cache_hit_tokens": self.prefix_cache_hit_tokens,
            "prefix_cache_hit_rate": round(cache_hit_rate, 6),
            "prefill_chunks": self.prefill_chunks,
            "decode_steps": self.decode_steps,
            "preemptions": self.preemptions,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": (
                round(self.accepted_tokens / self.proposed_tokens, 6)
                if self.proposed_tokens
                else None
            ),
        }
