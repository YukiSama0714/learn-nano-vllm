import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render nano-vLLM benchmark JSON files as a Markdown table."
    )
    parser.add_argument("results", type=Path, nargs="+")
    return parser.parse_args()


def format_number(value):
    return "-" if value is None else f"{value:.2f}"


def main():
    args = parse_args()
    headers = [
        "Result",
        "Policy",
        "Chunk",
        "TTFT P50 ms",
        "TTFT P95 ms",
        "TPOT P50 ms",
        "TPOT P95 ms",
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
        row = [
            path.stem,
            config["scheduling_policy"],
            str(config["prefill_chunk_size"]),
            format_number(summary["ttft_ms"]["p50"]),
            format_number(summary["ttft_ms"]["p95"]),
            format_number(summary["tpot_ms"]["p50"]),
            format_number(summary["tpot_ms"]["p95"]),
            format_number(summary["output_tokens_per_second"]["mean"]),
            format_number(summary["prefix_cache_hit_rate"]),
            format_number(summary["peak_reserved_gib"]),
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
