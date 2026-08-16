from dataclasses import dataclass


class NgramProposer:
    def __init__(self, min_size: int, max_size: int):
        if not 1 <= min_size <= max_size:
            raise ValueError("ngram bounds must satisfy 1 <= min <= max")
        self.min_size = min_size
        self.max_size = max_size

    def propose(self, token_ids: list[int], max_tokens: int) -> list[int]:
        if max_tokens <= 0:
            return []
        max_size = min(self.max_size, len(token_ids) - 1)
        for size in range(max_size, self.min_size - 1, -1):
            suffix = token_ids[-size:]
            for start in range(len(token_ids) - size - 1, -1, -1):
                if token_ids[start : start + size] != suffix:
                    continue
                proposal_start = start + size
                proposal_end = min(
                    len(token_ids),
                    proposal_start + max_tokens,
                )
                return token_ids[proposal_start:proposal_end]
        return []


@dataclass(slots=True)
class VerificationResult:
    output_token_ids: list[int]
    accepted_tokens: int


def verify_greedy_tokens(
    proposed_token_ids: list[int],
    target_token_ids: list[int],
) -> VerificationResult:
    expected_target_tokens = len(proposed_token_ids) + 1
    if len(target_token_ids) != expected_target_tokens:
        raise ValueError(
            "target verification must return one token per proposal plus "
            "one bonus token"
        )
    output_token_ids = []
    for index, proposed_token in enumerate(proposed_token_ids):
        target_token = target_token_ids[index]
        if proposed_token != target_token:
            output_token_ids.append(target_token)
            return VerificationResult(output_token_ids, index)
        output_token_ids.append(proposed_token)
    output_token_ids.append(target_token_ids[-1])
    return VerificationResult(output_token_ids, len(proposed_token_ids))
