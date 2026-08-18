from abc import ABC, abstractmethod

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import BatchMetadata, get_context

PAGED_ATTENTION_PARTITION_SIZE = 512
PAGED_ATTENTION_MIN_PROGRAMS = 1024


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    key = tl.load(key_ptr + key_offsets, mask=mask)
    value = tl.load(value_ptr + value_offsets, mask=mask)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


@triton.jit
def paged_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    query_to_request_ptr,
    query_positions_ptr,
    output_ptr,
    query_stride_token,
    query_stride_head,
    key_stride_block,
    key_stride_token,
    key_stride_head,
    value_stride_block,
    value_stride_token,
    value_stride_head,
    block_table_stride,
    output_stride_token,
    output_stride_head,
    num_query_heads,
    num_kv_heads,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_index = tl.program_id(0)
    query_head = tl.program_id(1)
    request_index = tl.load(query_to_request_ptr + query_index)
    query_position = tl.load(query_positions_ptr + query_index)
    context_length = query_position + 1
    kv_head = query_head // (num_query_heads // num_kv_heads)

    offsets_d = tl.arange(0, BLOCK_D)
    head_mask = offsets_d < HEAD_DIM
    query_offsets = (
        query_index * query_stride_token + query_head * query_stride_head + offsets_d
    )
    query = tl.load(query_ptr + query_offsets, mask=head_mask).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)
    for token_start in tl.range(0, context_length, BLOCK_N):
        token_indices = token_start + offsets_n
        token_mask = token_indices < context_length
        logical_blocks = token_indices // PAGE_SIZE
        block_offsets = token_indices % PAGE_SIZE
        physical_blocks = tl.load(
            block_tables_ptr + request_index * block_table_stride + logical_blocks,
            mask=token_mask,
            other=0,
        )
        key_offsets = (
            physical_blocks[:, None] * key_stride_block
            + block_offsets[:, None] * key_stride_token
            + kv_head * key_stride_head
            + offsets_d[None, :]
        )
        key = tl.load(
            key_cache_ptr + key_offsets,
            mask=token_mask[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        new_sum = running_sum * correction + tl.sum(probabilities, axis=0)

        value_offsets = (
            physical_blocks[:, None] * value_stride_block
            + block_offsets[:, None] * value_stride_token
            + kv_head * value_stride_head
            + offsets_d[None, :]
        )
        value = tl.load(
            value_cache_ptr + value_offsets,
            mask=token_mask[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * value,
            axis=0,
        )
        running_max = new_max
        running_sum = new_sum

    output_offsets = (
        query_index * output_stride_token + query_head * output_stride_head + offsets_d
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator / running_sum,
        mask=head_mask,
    )


@triton.jit
def paged_attention_splitk_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    query_to_request_ptr,
    query_positions_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    query_stride_token,
    query_stride_head,
    key_stride_block,
    key_stride_token,
    key_stride_head,
    value_stride_block,
    value_stride_token,
    value_stride_head,
    block_table_stride,
    partial_scalar_stride_token,
    partial_scalar_stride_head,
    partial_scalar_stride_partition,
    partial_acc_stride_token,
    partial_acc_stride_head,
    partial_acc_stride_partition,
    partial_acc_stride_dim,
    num_query_heads,
    num_kv_heads,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    query_index = tl.program_id(0)
    query_head = tl.program_id(1)
    partition_index = tl.program_id(2)
    request_index = tl.load(query_to_request_ptr + query_index)
    query_position = tl.load(query_positions_ptr + query_index)
    context_length = query_position + 1
    partition_start = partition_index * PARTITION_SIZE
    partition_end = tl.minimum(context_length, partition_start + PARTITION_SIZE)

    scalar_offset = (
        query_index * partial_scalar_stride_token
        + query_head * partial_scalar_stride_head
        + partition_index * partial_scalar_stride_partition
    )
    offsets_d = tl.arange(0, BLOCK_D)
    head_mask = offsets_d < HEAD_DIM
    acc_offsets = (
        query_index * partial_acc_stride_token
        + query_head * partial_acc_stride_head
        + partition_index * partial_acc_stride_partition
        + offsets_d * partial_acc_stride_dim
    )
    if partition_start >= context_length:
        tl.store(partial_max_ptr + scalar_offset, -float("inf"))
        tl.store(partial_sum_ptr + scalar_offset, 0.0)
        tl.store(partial_acc_ptr + acc_offsets, 0.0, mask=head_mask)
        return

    kv_head = query_head // (num_query_heads // num_kv_heads)
    query_offsets = (
        query_index * query_stride_token + query_head * query_stride_head + offsets_d
    )
    query = tl.load(query_ptr + query_offsets, mask=head_mask).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)
    for token_start in tl.range(partition_start, partition_end, BLOCK_N):
        token_indices = token_start + offsets_n
        token_mask = token_indices < partition_end
        logical_blocks = token_indices // PAGE_SIZE
        block_offsets = token_indices % PAGE_SIZE
        physical_blocks = tl.load(
            block_tables_ptr + request_index * block_table_stride + logical_blocks,
            mask=token_mask,
            other=0,
        )
        key_offsets = (
            physical_blocks[:, None] * key_stride_block
            + block_offsets[:, None] * key_stride_token
            + kv_head * key_stride_head
            + offsets_d[None, :]
        )
        key = tl.load(
            key_cache_ptr + key_offsets,
            mask=token_mask[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(key * query[None, :], axis=1) * softmax_scale
        scores = tl.where(token_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        new_sum = running_sum * correction + tl.sum(probabilities, axis=0)

        value_offsets = (
            physical_blocks[:, None] * value_stride_block
            + block_offsets[:, None] * value_stride_token
            + kv_head * value_stride_head
            + offsets_d[None, :]
        )
        value = tl.load(
            value_cache_ptr + value_offsets,
            mask=token_mask[:, None] & head_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * value,
            axis=0,
        )
        running_max = new_max
        running_sum = new_sum

    tl.store(partial_max_ptr + scalar_offset, running_max)
    tl.store(partial_sum_ptr + scalar_offset, running_sum)
    tl.store(partial_acc_ptr + acc_offsets, accumulator, mask=head_mask)


@triton.jit
def paged_attention_splitk_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    query_positions_ptr,
    output_ptr,
    partial_scalar_stride_token,
    partial_scalar_stride_head,
    partial_scalar_stride_partition,
    partial_acc_stride_token,
    partial_acc_stride_head,
    partial_acc_stride_partition,
    partial_acc_stride_dim,
    output_stride_token,
    output_stride_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PARTITIONS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    query_index = tl.program_id(0)
    query_head = tl.program_id(1)
    context_length = tl.load(query_positions_ptr + query_index) + 1
    num_partitions = (context_length + PARTITION_SIZE - 1) // PARTITION_SIZE

    offsets_p = tl.arange(0, BLOCK_PARTITIONS)
    partition_mask = offsets_p < num_partitions
    scalar_offsets = (
        query_index * partial_scalar_stride_token
        + query_head * partial_scalar_stride_head
        + offsets_p * partial_scalar_stride_partition
    )
    partial_max = tl.load(
        partial_max_ptr + scalar_offsets,
        mask=partition_mask,
        other=-float("inf"),
    )
    global_max = tl.max(partial_max, axis=0)
    correction = tl.exp(partial_max - global_max)
    partial_sum = tl.load(
        partial_sum_ptr + scalar_offsets,
        mask=partition_mask,
        other=0.0,
    )
    denominator = tl.sum(partial_sum * correction, axis=0)

    offsets_d = tl.arange(0, BLOCK_D)
    head_mask = offsets_d < HEAD_DIM
    acc_offsets = (
        query_index * partial_acc_stride_token
        + query_head * partial_acc_stride_head
        + offsets_p[:, None] * partial_acc_stride_partition
        + offsets_d[None, :] * partial_acc_stride_dim
    )
    partial_acc = tl.load(
        partial_acc_ptr + acc_offsets,
        mask=partition_mask[:, None] & head_mask[None, :],
        other=0.0,
    )
    accumulator = tl.sum(partial_acc * correction[:, None], axis=0)

    output_offsets = (
        query_index * output_stride_token + query_head * output_stride_head + offsets_d
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator / denominator,
        mask=head_mask,
    )


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    num_tokens, num_heads, head_dim = key.shape
    width = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == width and v_cache.stride(1) == width
    assert slot_mapping.numel() == num_tokens
    store_kvcache_kernel[(num_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        width,
        triton.next_power_of_2(width),
    )


def _num_splitk_partitions(block_tables: torch.Tensor, page_size: int) -> int:
    max_context_capacity = block_tables.shape[1] * page_size
    return max(
        1,
        (max_context_capacity + PAGED_ATTENTION_PARTITION_SIZE - 1)
        // PAGED_ATTENTION_PARTITION_SIZE,
    )


def allocate_splitk_workspace(
    query: torch.Tensor,
    num_partitions: int,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_tokens, num_query_heads, head_dim = query.shape
    num_tokens = query_tokens if num_tokens is None else num_tokens
    scalar_shape = (num_tokens, num_query_heads, num_partitions)
    partial_max = torch.empty(
        scalar_shape,
        dtype=torch.float32,
        device=query.device,
    )
    partial_sum = torch.empty_like(partial_max)
    partial_acc = torch.empty(
        *scalar_shape,
        head_dim,
        dtype=torch.float32,
        device=query.device,
    )
    return partial_max, partial_sum, partial_acc


def resolve_paged_attention_decode_kernel(
    decode_kernel: str,
    num_tokens: int,
    num_query_heads: int,
    num_partitions: int,
) -> str:
    if decode_kernel not in {"general", "split_k", "auto"}:
        raise ValueError(f"unknown paged-attention decode kernel: {decode_kernel}")
    if decode_kernel != "auto":
        return decode_kernel
    use_split_k = (
        num_partitions > 1
        and num_tokens * num_query_heads < PAGED_ATTENTION_MIN_PROGRAMS
    )
    return "split_k" if use_split_k else "general"


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: float,
    page_size: int,
    decode_kernel: str = "general",
    splitk_workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    num_tokens, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[2]
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if page_size not in {16, 32, 64}:
        raise ValueError("paged attention supports page sizes 16, 32, and 64")
    if key_cache.shape[1] != page_size or value_cache.shape[1] != page_size:
        raise ValueError("page size must match the KV cache layout")
    if block_tables.dtype != torch.int32:
        raise TypeError("block tables must use int32")
    output = torch.empty_like(query)
    block_d = triton.next_power_of_2(head_dim)
    num_partitions = _num_splitk_partitions(block_tables, page_size)
    resolved_kernel = resolve_paged_attention_decode_kernel(
        decode_kernel, num_tokens, num_query_heads, num_partitions
    )
    if resolved_kernel == "split_k":
        if splitk_workspace is None:
            splitk_workspace = allocate_splitk_workspace(query, num_partitions)
        partial_max, partial_sum, partial_acc = splitk_workspace
        scalar_capacity = partial_max.shape
        if (
            len(scalar_capacity) != 3
            or partial_sum.shape != scalar_capacity
            or scalar_capacity[0] < num_tokens
            or scalar_capacity[1] != num_query_heads
            or scalar_capacity[2] < num_partitions
            or partial_acc.shape != scalar_capacity + (head_dim,)
        ):
            raise ValueError("split-K workspace is too small for the decode batch")
        if any(
            tensor.dtype != torch.float32 or tensor.device != query.device
            for tensor in splitk_workspace
        ):
            raise ValueError("split-K workspace must be FP32 on the query device")
        paged_attention_splitk_kernel[(num_tokens, num_query_heads, num_partitions)](
            query,
            key_cache,
            value_cache,
            block_tables,
            query_to_request,
            query_positions,
            partial_max,
            partial_sum,
            partial_acc,
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_tables.stride(0),
            partial_max.stride(0),
            partial_max.stride(1),
            partial_max.stride(2),
            partial_acc.stride(0),
            partial_acc.stride(1),
            partial_acc.stride(2),
            partial_acc.stride(3),
            num_query_heads,
            num_kv_heads,
            softmax_scale,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            PAGE_SIZE=page_size,
            BLOCK_N=32,
            PARTITION_SIZE=PAGED_ATTENTION_PARTITION_SIZE,
            num_warps=4,
        )
        paged_attention_splitk_reduce_kernel[(num_tokens, num_query_heads)](
            partial_max,
            partial_sum,
            partial_acc,
            query_positions,
            output,
            partial_max.stride(0),
            partial_max.stride(1),
            partial_max.stride(2),
            partial_acc.stride(0),
            partial_acc.stride(1),
            partial_acc.stride(2),
            partial_acc.stride(3),
            output.stride(0),
            output.stride(1),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            BLOCK_PARTITIONS=triton.next_power_of_2(num_partitions),
            PARTITION_SIZE=PAGED_ATTENTION_PARTITION_SIZE,
            num_warps=4,
        )
        return output
    paged_attention_kernel[(num_tokens, num_query_heads)](
        query,
        key_cache,
        value_cache,
        block_tables,
        query_to_request,
        query_positions,
        output,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        output.stride(0),
        output.stride(1),
        num_query_heads,
        num_kv_heads,
        softmax_scale,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        BLOCK_N=32,
        num_warps=4,
    )
    return output


class AttentionBackend(ABC):
    @abstractmethod
    def forward(
        self,
        attention: "Attention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BatchMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def store(
        attention: "Attention",
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BatchMetadata,
    ):
        if attention.k_cache.numel() and attention.v_cache.numel():
            store_kvcache(
                key,
                value,
                attention.k_cache,
                attention.v_cache,
                metadata.slot_mapping,
            )


class FlashAttentionBackend(AttentionBackend):
    def forward(
        self,
        attention: "Attention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BatchMetadata,
    ) -> torch.Tensor:
        self.store(attention, key, value, metadata)
        if metadata.is_prefill:
            if metadata.block_tables is not None:
                key, value = attention.k_cache, attention.v_cache
            return flash_attn_varlen_func(
                query,
                key,
                value,
                max_seqlen_q=metadata.max_seqlen_q,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_k=metadata.max_seqlen_k,
                cu_seqlens_k=metadata.cu_seqlens_k,
                softmax_scale=attention.scale,
                causal=True,
                block_table=metadata.block_tables,
            )
        return flash_attn_with_kvcache(
            query.unsqueeze(1),
            attention.k_cache,
            attention.v_cache,
            cache_seqlens=metadata.context_lens,
            block_table=metadata.block_tables,
            softmax_scale=attention.scale,
            causal=True,
        )


class TritonPagedAttentionBackend(AttentionBackend):
    def forward(
        self,
        attention: "Attention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: BatchMetadata,
    ) -> torch.Tensor:
        self.store(attention, key, value, metadata)
        if metadata.block_tables is None:
            return flash_attn_varlen_func(
                query,
                key,
                value,
                max_seqlen_q=metadata.max_seqlen_q,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_k=metadata.max_seqlen_k,
                cu_seqlens_k=metadata.cu_seqlens_k,
                softmax_scale=attention.scale,
                causal=True,
            )
        decode_kernel = (
            attention.paged_attention_decode_kernel
            if not metadata.is_prefill
            else "general"
        )
        num_partitions = _num_splitk_partitions(
            metadata.block_tables,
            attention.block_size,
        )
        splitk_workspace = None
        resolved_kernel = resolve_paged_attention_decode_kernel(
            decode_kernel,
            query.shape[0],
            query.shape[1],
            num_partitions,
        )
        if resolved_kernel == "split_k":
            splitk_workspace = attention.get_splitk_workspace(
                query,
                num_partitions,
            )
        return paged_attention(
            query,
            attention.k_cache,
            attention.v_cache,
            metadata.block_tables,
            metadata.query_to_request,
            metadata.query_positions,
            attention.scale,
            attention.block_size,
            decode_kernel=decode_kernel,
            splitk_workspace=splitk_workspace,
        )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        backend: str = "flash_attn",
        block_size: int = 256,
        paged_attention_decode_kernel: str = "auto",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        if paged_attention_decode_kernel not in {"general", "split_k", "auto"}:
            raise ValueError(
                "paged-attention decode kernel must be general, split_k, or auto"
            )
        self.paged_attention_decode_kernel = paged_attention_decode_kernel
        self._splitk_workspace: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self.backend: AttentionBackend
        if backend == "flash_attn":
            self.backend = FlashAttentionBackend()
        elif backend == "triton_paged":
            self.backend = TritonPagedAttentionBackend()
        else:
            raise ValueError(f"unknown attention backend: {backend}")
        self.k_cache = self.v_cache = torch.tensor([])

    def get_splitk_workspace(
        self,
        query: torch.Tensor,
        num_partitions: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens, num_query_heads, head_dim = query.shape
        current = self._splitk_workspace
        if current is not None:
            scalar_shape = current[0].shape
            if (
                current[0].device == query.device
                and scalar_shape[0] >= num_tokens
                and scalar_shape[1] == num_query_heads
                and scalar_shape[2] >= num_partitions
                and current[2].shape[-1] == head_dim
            ):
                return current
            num_tokens = max(num_tokens, scalar_shape[0])
            num_partitions = max(num_partitions, scalar_shape[2])
        self._splitk_workspace = allocate_splitk_workspace(
            query,
            num_partitions,
            num_tokens=num_tokens,
        )
        return self._splitk_workspace

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self.backend.forward(self, query, key, value, get_context())
