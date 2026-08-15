import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark request-level nano-vLLM latency and throughput."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--scheduling-policy",
        choices=("prefill_first", "slo_aware"),
        default="prefill_first",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--max-consecutive-decode-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.shared_prefix_len <= args.input_len:
        parser.error("--shared-prefix-len must be between 0 and --input-len")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input and output lengths exceed --max-model-len")
    if args.num_requests <= 0 or args.repeats <= 0:
        parser.error("request and repeat counts must be positive")
    if not 0 <= args.prefill_chunk_size <= args.max_num_batched_tokens:
        parser.error(
            "--prefill-chunk-size must not exceed "
            "--max-num-batched-tokens"
        )
    if args.ttft_slo_ms < 0 or args.max_consecutive_decode_steps <= 0:
        parser.error("SLO scheduler parameters must be non-negative")
    return args


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(values, 0.5), 3),
        "p95": round(percentile(values, 0.95), 3),
    }


def make_workload(args, vocab_size, repeat):
    rng = random.Random(args.seed + repeat)
    shared_prefix = [
        rng.randrange(vocab_size) for _ in range(args.shared_prefix_len)
    ]
    suffix_len = args.input_len - args.shared_prefix_len
    prompts = [
        shared_prefix + [rng.randrange(vocab_size) for _ in range(suffix_len)]
        for _ in range(args.num_requests)
    ]
    cache_warmup_prompt = shared_prefix + [
        rng.randrange(vocab_size) for _ in range(suffix_len)
    ]
    return prompts, cache_warmup_prompt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    prefill_chunk_size = (
        args.prefill_chunk_size or args.max_num_batched_tokens
    )
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduling_policy=args.scheduling_policy,
        prefill_chunk_size=prefill_chunk_size,
        ttft_slo_ms=args.ttft_slo_ms,
        max_consecutive_decode_steps=args.max_consecutive_decode_steps,
    )
    sampling_params = SamplingParams(
        temperature=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    warmup_params = SamplingParams(
        temperature=1.0,
        ignore_eos=True,
        max_tokens=8,
    )
    llm.generate([[1, 2, 3, 4]], warmup_params, use_tqdm=False)
    torch.cuda.synchronize()

    runs = []
    request_metrics = []
    for repeat in range(args.repeats):
        prompts, cache_warmup_prompt = make_workload(
            args,
            llm.tokenizer.vocab_size,
            repeat,
        )
        if args.shared_prefix_len:
            llm.generate(
                [cache_warmup_prompt],
                SamplingParams(
                    temperature=1.0,
                    ignore_eos=True,
                    max_tokens=1,
                ),
                use_tqdm=False,
            )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at

        metrics = [output["metrics"] for output in outputs]
        request_metrics.extend(metrics)
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        runs.append(
            {
                "repeat": repeat,
                "elapsed_ms": round(elapsed * 1000, 3),
                "requests_per_second": round(args.num_requests / elapsed, 3),
                "output_tokens_per_second": round(output_tokens / elapsed, 3),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 1024**3,
                    3,
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 1024**3,
                    3,
                ),
            }
        )

    total_prompt_tokens = sum(
        metric["prompt_tokens"] for metric in request_metrics
    )
    total_cache_hit_tokens = sum(
        metric["prefix_cache_hit_tokens"] for metric in request_metrics
    )
    result = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": vars(args)
        | {
            "output": str(args.output) if args.output else None,
            "prefill_chunk_size": prefill_chunk_size,
        },
        "summary": {
            "elapsed_ms": summarize([run["elapsed_ms"] for run in runs]),
            "requests_per_second": summarize(
                [run["requests_per_second"] for run in runs]
            ),
            "output_tokens_per_second": summarize(
                [run["output_tokens_per_second"] for run in runs]
            ),
            "ttft_ms": summarize(
                [metric["ttft_ms"] for metric in request_metrics]
            ),
            "tpot_ms": summarize(
                [metric["tpot_ms"] for metric in request_metrics]
            ),
            "e2e_ms": summarize(
                [metric["e2e_ms"] for metric in request_metrics]
            ),
            "queue_ms": summarize(
                [metric["queue_ms"] for metric in request_metrics]
            ),
            "prefill_ms": summarize(
                [metric["prefill_ms"] for metric in request_metrics]
            ),
            "decode_ms": summarize(
                [metric["decode_ms"] for metric in request_metrics]
            ),
            "prefill_chunks": summarize(
                [metric["prefill_chunks"] for metric in request_metrics]
            ),
            "decode_steps": summarize(
                [metric["decode_steps"] for metric in request_metrics]
            ),
            "preemptions": summarize(
                [metric["preemptions"] for metric in request_metrics]
            ),
            "prefix_cache_hit_rate": round(
                total_cache_hit_tokens / total_prompt_tokens,
                6,
            ),
            "peak_allocated_gib": max(
                run["peak_allocated_gib"] for run in runs
            ),
            "peak_reserved_gib": max(
                run["peak_reserved_gib"] for run in runs
            ),
        },
        "runs": runs,
        "request_metrics": request_metrics,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
