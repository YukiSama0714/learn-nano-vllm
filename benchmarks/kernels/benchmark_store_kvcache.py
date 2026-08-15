import argparse
import json
from pathlib import Path

import torch
import triton

from nanovllm.layers.attention import store_kvcache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and benchmark the Triton KV-cache store kernel."
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 8, 64, 512, 4096],
    )
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def pytorch_store(key, value, k_cache, v_cache, slot_mapping):
    flat_k_cache = k_cache.view(-1, key.size(1), key.size(2))
    flat_v_cache = v_cache.view(-1, value.size(1), value.size(2))
    flat_k_cache[slot_mapping.long()] = key
    flat_v_cache[slot_mapping.long()] = value


def benchmark_case(args, num_tokens):
    dtype = torch.bfloat16
    num_slots = (
        (num_tokens + args.block_size - 1) // args.block_size + 1
    ) * args.block_size
    num_blocks = num_slots // args.block_size
    key = torch.randn(
        num_tokens,
        args.num_kv_heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    value = torch.randn_like(key)
    k_cache = torch.zeros(
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.randperm(num_slots, device="cuda")[:num_tokens].int()

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize()
    triton_k_cache = k_cache.clone()
    triton_v_cache = v_cache.clone()

    k_cache.zero_()
    v_cache.zero_()
    pytorch_store(key, value, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize()
    torch.testing.assert_close(triton_k_cache, k_cache, rtol=0, atol=0)
    torch.testing.assert_close(triton_v_cache, v_cache, rtol=0, atol=0)

    triton_ms = triton.testing.do_bench(
        lambda: store_kvcache(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    pytorch_ms = triton.testing.do_bench(
        lambda: pytorch_store(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    transferred_bytes = 4 * key.numel() * key.element_size()
    return {
        "num_tokens": num_tokens,
        "triton_ms": round(triton_ms, 6),
        "pytorch_ms": round(pytorch_ms, 6),
        "speedup": round(pytorch_ms / triton_ms, 3),
        "triton_effective_gbps": round(
            transferred_bytes / triton_ms / 1e6,
            3,
        ),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    results = [
        benchmark_case(args, num_tokens) for num_tokens in args.num_tokens
    ]
    report = {
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": {
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
            "warmup": args.warmup,
            "rep": args.rep,
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
