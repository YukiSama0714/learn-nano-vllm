from dataclasses import dataclass, field

from nanovllm.engine.sequence import Sequence


@dataclass(slots=True)
class ScheduledRequest:
    sequence: Sequence
    num_scheduled_tokens: int
    is_prefill: bool
    needs_sampling: bool
    speculative_token_ids: list[int] = field(default_factory=list)
    speculative_method: str = "none"
    accepted_tokens: int = 0


@dataclass(slots=True)
class SchedulerOutput:
    scheduled_requests: list[ScheduledRequest]

    @classmethod
    def from_phase(
        cls,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> "SchedulerOutput":
        return cls(
            [
                ScheduledRequest(
                    sequence=seq,
                    num_scheduled_tokens=seq.num_scheduled_tokens,
                    is_prefill=is_prefill,
                    needs_sampling=(
                        not is_prefill
                        or seq.num_cached_tokens + seq.num_scheduled_tokens
                        == seq.num_tokens
                    ),
                )
                for seq in sequences
            ]
        )

    @property
    def sequences(self) -> list[Sequence]:
        return [request.sequence for request in self.scheduled_requests]

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(request.num_scheduled_tokens for request in self.scheduled_requests)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(
            request.num_scheduled_tokens
            for request in self.scheduled_requests
            if request.is_prefill
        )

    @property
    def num_decode_tokens(self) -> int:
        return sum(
            request.num_scheduled_tokens
            for request in self.scheduled_requests
            if not request.is_prefill
        )

    @property
    def num_proposed_tokens(self) -> int:
        return sum(
            len(request.speculative_token_ids) for request in self.scheduled_requests
        )

    @property
    def num_accepted_tokens(self) -> int:
        return sum(request.accepted_tokens for request in self.scheduled_requests)

    @property
    def is_mixed(self) -> bool:
        return self.num_prefill_tokens > 0 and self.num_decode_tokens > 0

    @property
    def is_prefill_only(self) -> bool:
        return self.num_prefill_tokens > 0 and self.num_decode_tokens == 0

    @property
    def is_decode_only(self) -> bool:
        return self.num_decode_tokens > 0 and self.num_prefill_tokens == 0

    @property
    def legacy_is_prefill(self) -> bool | None:
        if self.is_prefill_only:
            return True
        if self.is_decode_only:
            return False
        return None

    def __iter__(self):
        """Keep phase-exclusive scheduler callers source compatible."""
        yield self.sequences
        yield self.legacy_is_prefill


@dataclass(slots=True)
class ModelRunnerOutput:
    token_ids: list[int | list[int] | None]
    input_prep_seconds: float = 0.0
    model_seconds: float = 0.0
    sampling_seconds: float = 0.0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
