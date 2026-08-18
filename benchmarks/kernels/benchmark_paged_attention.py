import argparse
import json
from functools import partial
from pathlib import Path

import torch

from nanovllm.layers.attention import (
    PAGED_ATTENTION_PARTITION_SIZE,
    allocate_splitk_workspace,
    paged_attention,
    resolve_paged_attention_decode_kernel,
)


def pytorch_reference(
    query,
    key_cache,
    value_cache,
    block_tables,
    context_length,
    scale,
):
    page_size = key_cache.shape[1]
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    heads_per_kv = num_query_heads // num_kv_heads
    logical_tokens = torch.arange(context_length, device=query.device)
    logical_blocks = logical_tokens // page_size
    block_offsets = logical_tokens % page_size
    outputs = []
    for request_index in range(query.shape[0]):
        physical_blocks = block_tables[request_index, logical_blocks]
        key = key_cache[physical_blocks, block_offsets]
        value = value_cache[physical_blocks, block_offsets]
        key = key.repeat_interleave(heads_per_kv, dim=1).float()
        value = value.repeat_interleave(heads_per_kv, dim=1).float()
        scores = (
            torch.einsum(
                "hd,thd->ht",
                query[request_index].float(),
                key,
            )
            * scale
        )
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, value))
    return torch.stack(outputs).to(query.dtype)


def benchmark(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the Triton paged-attention decode kernel."
    )
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[128, 512, 2048, 4096],
    )
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--decode-kernels",
        nargs="+",
        choices=("general", "split_k", "auto"),
        default=("general", "split_k"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(page_size not in {16, 32, 64} for page_size in args.page_sizes):
        parser.error("page sizes must be 16, 32, or 64")
    if args.num_query_heads % args.num_kv_heads:
        parser.error("query heads must be divisible by KV heads")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = []
    scale = args.head_dim**-0.5
    for page_size in args.page_sizes:
        for batch_size in args.batch_sizes:
            for context_length in args.context_lengths:
                pages_per_request = (context_length + page_size - 1) // page_size
                num_blocks = batch_size * pages_per_request
                key_cache = torch.randn(
                    num_blocks,
                    page_size,
                    args.num_kv_heads,
                    args.head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                value_cache = torch.randn_like(key_cache)
                block_tables = torch.arange(
                    num_blocks,
                    device="cuda",
                    dtype=torch.int32,
                ).view(batch_size, pages_per_request)
                query = torch.randn(
                    batch_size,
                    args.num_query_heads,
                    args.head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                query_to_request = torch.arange(
                    batch_size,
                    device="cuda",
                    dtype=torch.int32,
                )
                query_positions = torch.full(
                    (batch_size,),
                    context_length - 1,
                    device="cuda",
                    dtype=torch.int32,
                )

                expected = pytorch_reference(
                    query,
                    key_cache,
                    value_cache,
                    block_tables,
                    context_length,
                    scale,
                )
                logical_bytes = (
                    2
                    * batch_size
                    * context_length
                    * args.num_query_heads
                    * args.head_dim
                    * query.element_size()
                )
                num_partitions = (
                    context_length + PAGED_ATTENTION_PARTITION_SIZE - 1
                ) // PAGED_ATTENTION_PARTITION_SIZE
                workspace = allocate_splitk_workspace(
                    query,
                    num_partitions,
                )
                for decode_kernel in args.decode_kernels:
                    resolved_kernel = resolve_paged_attention_decode_kernel(
                        decode_kernel,
                        batch_size,
                        args.num_query_heads,
                        num_partitions,
                    )
                    run_triton = partial(
                        paged_attention,
                        query,
                        key_cache,
                        value_cache,
                        block_tables,
                        query_to_request,
                        query_positions,
                        scale,
                        page_size,
                        decode_kernel,
                        workspace,
                    )
                    actual = run_triton()
                    max_error = (actual - expected).abs().max().item()
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=2e-2,
                        atol=2e-2,
                    )
                    triton_ms = benchmark(
                        run_triton,
                        args.warmup,
                        args.repeats,
                    )
                    rows.append(
                        {
                            "decode_kernel": decode_kernel,
                            "resolved_kernel": resolved_kernel,
                            "page_size": page_size,
                            "batch_size": batch_size,
                            "context_length": context_length,
                            "triton_ms": round(triton_ms, 6),
                            "effective_gb_per_second": round(
                                logical_bytes / triton_ms / 1e6,
                                3,
                            ),
                            "max_error": max_error,
                        }
                    )
                    del run_triton, actual
                del (
                    query,
                    key_cache,
                    value_cache,
                    block_tables,
                    query_to_request,
                    query_positions,
                    expected,
                    workspace,
                )
    general_ms = {
        (row["page_size"], row["batch_size"], row["context_length"]): row["triton_ms"]
        for row in rows
        if row["decode_kernel"] == "general"
    }
    for row in rows:
        baseline_ms = general_ms.get(
            (row["page_size"], row["batch_size"], row["context_length"])
        )
        row["speedup_vs_general"] = (
            round(baseline_ms / row["triton_ms"], 3)
            if baseline_ms is not None
            else None
        )
    result = {
        "schema_version": 1,
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "results": rows,
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
