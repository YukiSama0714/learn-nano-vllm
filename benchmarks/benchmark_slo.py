import argparse
import importlib.metadata
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from nanovllm import SamplingParams


def pytorch_store_kvcache(
    key,
    value,
    k_cache,
    v_cache,
    slot_mapping,
):
    flat_k_cache = k_cache.view(-1, key.size(1), key.size(2))
    flat_v_cache = v_cache.view(-1, value.size(1), value.size(2))
    indices = slot_mapping.long()
    flat_k_cache[indices] = key
    flat_v_cache[indices] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark request-level nano-vLLM latency and throughput."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-len", type=int)
    input_group.add_argument(
        "--input-lens",
        type=lambda value: [int(item) for item in value.split(",")],
    )
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--shared-prefix-len", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--scheduling-policy",
        choices=(
            "prefill_first",
            "slo_aware",
            "slo_aware_v2",
            "slo_aware_v3",
        ),
        default="prefill_first",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--prefill-chunk-granularity", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=50.0)
    parser.add_argument("--max-consecutive-decode-steps", type=int, default=8)
    parser.add_argument("--scheduler-cost-ema-alpha", type=float, default=0.2)
    parser.add_argument(
        "--attention-backend",
        choices=("flash_attn", "triton_paged"),
        default="flash_attn",
    )
    parser.add_argument(
        "--paged-attention-decode-kernel",
        choices=("general", "split_k", "auto"),
        default="auto",
    )
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument(
        "--speculative-method",
        choices=("none", "ngram", "draft"),
        default="none",
    )
    parser.add_argument("--draft-model")
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--record-token-diagnostics", action="store_true")
    parser.add_argument(
        "--kv-store-backend",
        choices=("triton", "pytorch"),
        default="triton",
    )
    parser.add_argument(
        "--rms-norm-backend",
        choices=("eager", "compiled", "triton"),
        default="compiled",
    )
    parser.add_argument(
        "--arrival-pattern",
        choices=("bulk", "constant", "poisson"),
        default="bulk",
    )
    parser.add_argument("--request-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input_len is None and args.input_lens is None:
        args.input_len = 1024
    input_lens = args.input_lens or [args.input_len]
    if not input_lens or any(length <= 0 for length in input_lens):
        parser.error("input lengths must be positive")
    if not 0 <= args.shared_prefix_len <= min(input_lens):
        parser.error("--shared-prefix-len must not exceed any input length")
    if max(input_lens) + args.output_len > args.max_model_len:
        parser.error("input and output lengths exceed --max-model-len")
    if args.num_requests <= 0 or args.repeats <= 0:
        parser.error("request and repeat counts must be positive")
    if not 0 <= args.prefill_chunk_size <= args.max_num_batched_tokens:
        parser.error("--prefill-chunk-size must not exceed --max-num-batched-tokens")
    resolved_chunk_size = args.prefill_chunk_size or args.max_num_batched_tokens
    if not 0 <= args.prefill_chunk_granularity <= resolved_chunk_size:
        parser.error("--prefill-chunk-granularity must not exceed the chunk size")
    if (
        args.ttft_slo_ms < 0
        or args.tpot_slo_ms <= 0
        or args.max_consecutive_decode_steps <= 0
    ):
        parser.error(
            "TTFT SLO must be non-negative; TPOT SLO and decode steps must be positive"
        )
    if not 0 < args.scheduler_cost_ema_alpha <= 1:
        parser.error("--scheduler-cost-ema-alpha must be in (0, 1]")
    if args.arrival_pattern != "bulk" and args.request_rate <= 0:
        parser.error("--request-rate must be positive for online arrivals")
    if args.kv_store_backend == "pytorch" and args.shared_prefix_len:
        parser.error("the PyTorch KV-store baseline does not support prefix-cache hits")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.speculative_method != "none" and args.temperature != 0:
        parser.error("speculative decoding MVP requires --temperature 0")
    if args.speculative_method == "draft" and not args.draft_model:
        parser.error("--draft-model is required for draft speculation")
    if args.speculative_method != "none" and args.scheduling_policy != "slo_aware_v3":
        parser.error("speculative decoding requires --scheduling-policy slo_aware_v3")
    if args.num_speculative_tokens <= 0:
        parser.error("--num-speculative-tokens must be positive")
    if not 1 <= args.ngram_min <= args.ngram_max:
        parser.error("ngram bounds must satisfy 1 <= min <= max")
    if args.attention_backend == "flash_attn" and args.kvcache_block_size % 256:
        parser.error("flash_attn requires a KV block size divisible by 256")
    if args.attention_backend == "triton_paged" and args.kvcache_block_size not in {
        16,
        32,
        64,
    }:
        parser.error("triton_paged requires a KV block size of 16, 32, or 64")
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


def violation_rate(values, target):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(value > target for value in values) / len(values), 6)


def make_arrival_offsets(pattern, request_rate, num_requests, rng):
    if pattern == "bulk":
        return [0.0] * num_requests
    if pattern == "constant":
        return [index / request_rate for index in range(num_requests)]
    offsets = [0.0]
    for _ in range(1, num_requests):
        offsets.append(offsets[-1] + rng.expovariate(request_rate))
    return offsets


def make_workload(args, vocab_size, repeat):
    rng = random.Random(args.seed + repeat)
    shared_prefix = [rng.randrange(vocab_size) for _ in range(args.shared_prefix_len)]
    input_lens = args.input_lens or [args.input_len]
    request_input_lens = [
        input_lens[index % len(input_lens)] for index in range(args.num_requests)
    ]
    rng.shuffle(request_input_lens)
    prompts = [
        shared_prefix
        + [rng.randrange(vocab_size) for _ in range(input_len - args.shared_prefix_len)]
        for input_len in request_input_lens
    ]
    cache_warmup_prompt = shared_prefix + [
        rng.randrange(vocab_size)
        for _ in range(max(input_lens) - args.shared_prefix_len)
    ]
    arrival_rng = random.Random(args.seed + 10_000 + repeat)
    arrival_offsets = make_arrival_offsets(
        args.arrival_pattern,
        args.request_rate,
        args.num_requests,
        arrival_rng,
    )
    return prompts, cache_warmup_prompt, arrival_offsets


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_config_summary(config):
    if config is None:
        return None
    return {
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "dtype": str(getattr(config, "dtype", None)),
        "max_position_embeddings": getattr(
            config,
            "max_position_embeddings",
            None,
        ),
    }


def summarize_step_metrics(step_metrics):
    if not step_metrics:
        return {
            "num_steps": 0,
            "mixed_steps": 0,
            "mixed_step_rate": 0.0,
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "acceptance_rate": None,
        }
    proposed = sum(step["proposed_tokens"] for step in step_metrics)
    accepted = sum(step["accepted_tokens"] for step in step_metrics)
    mixed = sum(step["mixed"] for step in step_metrics)
    result = {
        "num_steps": len(step_metrics),
        "mixed_steps": mixed,
        "mixed_step_rate": round(mixed / len(step_metrics), 6),
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": (round(accepted / proposed, 6) if proposed else None),
    }
    for metric in (
        "scheduler_ms",
        "input_prep_ms",
        "model_ms",
        "sampling_ms",
        "prefill_tokens",
        "decode_tokens",
        "kv_allocated_tokens",
        "kv_reserved_tokens",
        "kv_computed_tokens",
        "kv_uncomputed_tokens",
        "kv_tail_waste_tokens",
        "kv_block_utilization",
    ):
        result[metric] = summarize([step[metric] for step in step_metrics])
    return result


def run_workload(
    llm,
    prompts,
    sampling_params,
    arrival_offsets,
    clock=None,
    sleeper=None,
    synchronize=None,
):
    if len(prompts) != len(arrival_offsets):
        raise ValueError("prompts and arrival offsets must have equal lengths")
    if any(offset < 0 for offset in arrival_offsets):
        raise ValueError("arrival offsets must be non-negative")
    if arrival_offsets != sorted(arrival_offsets):
        raise ValueError("arrival offsets must be monotonic")
    clock = clock or time.perf_counter
    sleeper = sleeper or time.sleep
    started_at = clock()
    pending = 0
    outputs = {}
    while pending < len(prompts) or not llm.is_finished():
        now = clock()
        elapsed = now - started_at
        while pending < len(prompts) and arrival_offsets[pending] <= elapsed:
            arrival_time = started_at + arrival_offsets[pending]
            seq_id = llm.add_request(
                prompts[pending],
                sampling_params,
                arrival_time=arrival_time,
            )
            outputs[seq_id] = None
            pending += 1
        if not llm.is_finished():
            finished, _ = llm.step()
            for seq_id, token_ids in finished:
                outputs[seq_id] = {
                    "token_ids": token_ids,
                    "metrics": llm.take_finished_request_metrics(seq_id),
                }
        elif pending < len(prompts):
            delay = arrival_offsets[pending] - elapsed
            sleeper(min(max(delay, 0.0), 0.01))
    if synchronize is not None:
        synchronize()
    ordered_outputs = []
    for seq_id in sorted(outputs):
        output = outputs[seq_id]
        output["text"] = llm.tokenizer.decode(output["token_ids"])
        ordered_outputs.append(output)
    elapsed = clock() - started_at
    return ordered_outputs, elapsed


def main():
    from nanovllm import LLM

    args = parse_args()
    if args.kv_store_backend == "pytorch":
        from nanovllm.layers import attention

        attention.store_kvcache = pytorch_store_kvcache
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    prefill_chunk_size = args.prefill_chunk_size or args.max_num_batched_tokens
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
        prefill_chunk_granularity=args.prefill_chunk_granularity,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        max_consecutive_decode_steps=args.max_consecutive_decode_steps,
        scheduler_cost_ema_alpha=args.scheduler_cost_ema_alpha,
        rms_norm_backend=args.rms_norm_backend,
        attention_backend=args.attention_backend,
        paged_attention_decode_kernel=args.paged_attention_decode_kernel,
        kvcache_block_size=args.kvcache_block_size,
        speculative_method=args.speculative_method,
        draft_model=args.draft_model,
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        record_token_diagnostics=args.record_token_diagnostics,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    warmup_params = SamplingParams(
        temperature=(0 if args.speculative_method != "none" else 1.0),
        ignore_eos=True,
        max_tokens=8,
    )
    llm.generate([[1, 2, 3, 4]], warmup_params, use_tqdm=False)
    torch.cuda.synchronize()
    llm.scheduler.reset_cost_estimates()
    llm.reset_step_metrics()

    runs = []
    request_metrics = []
    all_step_metrics = []
    for repeat in range(args.repeats):
        prompts, cache_warmup_prompt, arrival_offsets = make_workload(
            args,
            llm.tokenizer.vocab_size,
            repeat,
        )
        if args.shared_prefix_len:
            llm.generate(
                [cache_warmup_prompt],
                SamplingParams(
                    temperature=(0 if args.speculative_method != "none" else 1.0),
                    ignore_eos=True,
                    max_tokens=1,
                ),
                use_tqdm=False,
            )

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        outputs, elapsed = run_workload(
            llm,
            prompts,
            sampling_params,
            arrival_offsets,
            synchronize=torch.cuda.synchronize,
        )
        step_metrics = llm.take_step_metrics()
        all_step_metrics.extend(step_metrics)

        token_diagnostics = [
            output["metrics"].pop("token_diagnostics", []) for output in outputs
        ]
        metrics = [output["metrics"] for output in outputs]
        request_metrics.extend(metrics)
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        run = {
            "repeat": repeat,
            "elapsed_ms": round(elapsed * 1000, 3),
            "requests_per_second": round(args.num_requests / elapsed, 3),
            "output_tokens_per_second": round(output_tokens / elapsed, 3),
            "arrival_span_ms": round(arrival_offsets[-1] * 1000, 3),
            "estimated_prefill_ms_per_token": round(
                (llm.scheduler.prefill_seconds_per_token or 0.0) * 1000,
                6,
            ),
            "estimated_decode_step_ms": round(
                (llm.scheduler.decode_step_seconds or 0.0) * 1000,
                3,
            ),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3,
                3,
            ),
            "steps": summarize_step_metrics(step_metrics),
            "output_token_ids": [output["token_ids"] for output in outputs],
        }
        if args.record_token_diagnostics:
            run["output_token_diagnostics"] = token_diagnostics
        runs.append(run)

    total_prompt_tokens = sum(metric["prompt_tokens"] for metric in request_metrics)
    total_cache_hit_tokens = sum(
        metric["prefix_cache_hit_tokens"] for metric in request_metrics
    )
    result = {
        "schema_version": 3,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": package_version("transformers"),
            "triton": package_version("triton"),
            "flash_attn": package_version("flash-attn"),
            "git_commit": git_commit(),
            "target_model_config": model_config_summary(llm.config.hf_config),
            "draft_model_config": model_config_summary(llm.config.draft_hf_config),
        },
        "config": vars(args)
        | {
            "output": str(args.output) if args.output else None,
            "prefill_chunk_size": prefill_chunk_size,
            "prefill_chunk_granularity": llm.config.prefill_chunk_granularity,
        },
        "summary": {
            "elapsed_ms": summarize([run["elapsed_ms"] for run in runs]),
            "requests_per_second": summarize(
                [run["requests_per_second"] for run in runs]
            ),
            "output_tokens_per_second": summarize(
                [run["output_tokens_per_second"] for run in runs]
            ),
            "ttft_ms": summarize([metric["ttft_ms"] for metric in request_metrics]),
            "tpot_ms": summarize([metric["tpot_ms"] for metric in request_metrics]),
            "e2e_ms": summarize([metric["e2e_ms"] for metric in request_metrics]),
            "queue_ms": summarize([metric["queue_ms"] for metric in request_metrics]),
            "prefill_ms": summarize(
                [metric["prefill_ms"] for metric in request_metrics]
            ),
            "decode_ms": summarize([metric["decode_ms"] for metric in request_metrics]),
            "inter_token_gap_p95_ms": summarize(
                [metric["inter_token_gap_p95_ms"] for metric in request_metrics]
            ),
            "max_inter_token_gap_ms": summarize(
                [metric["max_inter_token_gap_ms"] for metric in request_metrics]
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
            "ttft_slo_violation_rate": violation_rate(
                [metric["ttft_ms"] for metric in request_metrics],
                args.ttft_slo_ms,
            ),
            "tpot_slo_violation_rate": violation_rate(
                [metric["tpot_ms"] for metric in request_metrics],
                args.tpot_slo_ms,
            ),
            "inter_token_slo_violation_rate": violation_rate(
                [metric["max_inter_token_gap_ms"] for metric in request_metrics],
                args.tpot_slo_ms,
            ),
            "peak_allocated_gib": max(run["peak_allocated_gib"] for run in runs),
            "peak_reserved_gib": max(run["peak_reserved_gib"] for run in runs),
            "steps": summarize_step_metrics(all_step_metrics),
        },
        "runs": runs,
        "request_metrics": request_metrics,
        "step_metrics": all_step_metrics,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
