import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render nano-vLLM benchmark JSON files as a Markdown table."
    )
    parser.add_argument("--eval-ttft-slo-ms", type=float)
    parser.add_argument("--eval-itl-slo-ms", type=float)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    if args.eval_ttft_slo_ms is not None and args.eval_ttft_slo_ms < 0:
        parser.error("--eval-ttft-slo-ms must be non-negative")
    if args.eval_itl_slo_ms is not None and args.eval_itl_slo_ms <= 0:
        parser.error("--eval-itl-slo-ms must be positive")
    return args


def format_number(value):
    return "-" if value is None else f"{value:.2f}"


def summary_percentile(summary, name, percentile):
    value = summary.get(name)
    return None if value is None else value.get(percentile)


def format_rate(value):
    return "-" if value is None else f"{value * 100:.1f}%"


def request_violation_rate(request_metrics, metric, target):
    values = [
        request[metric]
        for request in request_metrics
        if request.get(metric) is not None
    ]
    if not values:
        return None
    return sum(value > target for value in values) / len(values)


def main():
    args = parse_args()
    ttft_violation_header = "TTFT violations"
    itl_violation_header = "Requests with ITL violation"
    if args.eval_ttft_slo_ms is not None:
        ttft_violation_header += f" > {args.eval_ttft_slo_ms:g}ms"
    if args.eval_itl_slo_ms is not None:
        itl_violation_header += f" > {args.eval_itl_slo_ms:g}ms"
    headers = [
        "Result",
        "Policy",
        "KV store",
        "Arrival",
        "Req/s",
        "Chunk",
        "TTFT/TPOT SLO ms",
        "TTFT P95 ms",
        "TPOT P95 ms",
        "ITL P95 ms",
        "Max ITL P95 ms",
        "Queue P95 ms",
        "Chunks P95",
        "E2E P95 ms",
        ttft_violation_header,
        itl_violation_header,
        "Output tok/s",
        "Cache hit",
        "Peak GiB",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        summary = result["summary"]
        ttft_violation_rate = summary.get("ttft_slo_violation_rate")
        itl_violation_rate = summary.get("inter_token_slo_violation_rate")
        if args.eval_ttft_slo_ms is not None:
            ttft_violation_rate = request_violation_rate(
                result["request_metrics"],
                "ttft_ms",
                args.eval_ttft_slo_ms,
            )
        if args.eval_itl_slo_ms is not None:
            itl_violation_rate = request_violation_rate(
                result["request_metrics"],
                "max_inter_token_gap_ms",
                args.eval_itl_slo_ms,
            )
        row = [
            path.stem,
            config["scheduling_policy"],
            config.get("kv_store_backend", "triton"),
            config.get("arrival_pattern", "bulk"),
            format_number(config.get("request_rate")),
            str(config["prefill_chunk_size"]),
            (
                f"{config.get('ttft_slo_ms', '-')}/"
                f"{config.get('tpot_slo_ms', '-')}"
            ),
            format_number(summary_percentile(summary, "ttft_ms", "p95")),
            format_number(summary_percentile(summary, "tpot_ms", "p95")),
            format_number(
                summary_percentile(
                    summary,
                    "inter_token_gap_p95_ms",
                    "p95",
                )
            ),
            format_number(
                summary_percentile(
                    summary,
                    "max_inter_token_gap_ms",
                    "p95",
                )
            ),
            format_number(summary_percentile(summary, "queue_ms", "p95")),
            format_number(
                summary_percentile(summary, "prefill_chunks", "p95")
            ),
            format_number(summary_percentile(summary, "e2e_ms", "p95")),
            format_rate(ttft_violation_rate),
            format_rate(itl_violation_rate),
            format_number(summary["output_tokens_per_second"]["mean"]),
            format_number(summary["prefix_cache_hit_rate"]),
            format_number(summary["peak_reserved_gib"]),
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
