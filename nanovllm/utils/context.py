from dataclasses import dataclass

import torch


@dataclass(slots=True)
class BatchMetadata:
    is_prefill: bool = False
    is_mixed: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    query_to_request: torch.Tensor | None = None
    query_positions: torch.Tensor | None = None

    @property
    def query_start_locs(self) -> torch.Tensor | None:
        return self.cu_seqlens_q

    @property
    def key_start_locs(self) -> torch.Tensor | None:
        return self.cu_seqlens_k


# Context remains an alias so external code written against nano-vLLM v1/v2
# does not need to change.
Context = BatchMetadata
_CONTEXT = BatchMetadata()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    logits_indices=None,
    is_mixed=False,
    query_to_request=None,
    query_positions=None,
):
    global _CONTEXT
    _CONTEXT = BatchMetadata(
        is_prefill=is_prefill,
        is_mixed=is_mixed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        logits_indices=logits_indices,
        query_to_request=query_to_request,
        query_positions=query_positions,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = BatchMetadata()
