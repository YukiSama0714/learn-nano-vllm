import argparse
import json
from pathlib import Path


def load_token_ids(path: Path) -> list[list[list[int]]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    runs = result.get("runs", [])
    if not runs or any("output_token_ids" not in run for run in runs):
        raise ValueError(
            f"{path} has no golden token IDs; rerun benchmark_slo.py "
            "with schema_version >= 2"
        )
    return [run["output_token_ids"] for run in runs]


def compare_token_ids(
    expected: list[list[list[int]]],
    actual: list[list[list[int]]],
) -> list[str]:
    mismatches = []
    if len(expected) != len(actual):
        return [f"repeat count differs: {len(expected)} != {len(actual)}"]
    for repeat_index, (expected_run, actual_run) in enumerate(zip(expected, actual)):
        if len(expected_run) != len(actual_run):
            mismatches.append(
                f"repeat {repeat_index}: request count differs: "
                f"{len(expected_run)} != {len(actual_run)}"
            )
            continue
        for request_index, (expected_tokens, actual_tokens) in enumerate(
            zip(expected_run, actual_run)
        ):
            if expected_tokens == actual_tokens:
                continue
            mismatch_index = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(expected_tokens, actual_tokens)
                    )
                    if left != right
                ),
                min(len(expected_tokens), len(actual_tokens)),
            )
            mismatches.append(
                f"repeat {repeat_index}, request {request_index}, "
                f"token {mismatch_index}: outputs differ"
            )
    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Compare greedy benchmark outputs token by token."
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    mismatches = compare_token_ids(
        load_token_ids(args.expected),
        load_token_ids(args.actual),
    )
    if mismatches:
        print("FAIL: greedy token IDs differ")
        for mismatch in mismatches[:20]:
            print(f"- {mismatch}")
        raise SystemExit(1)
    print("PASS: all greedy token IDs are identical")


if __name__ == "__main__":
    main()
