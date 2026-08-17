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


def load_token_diagnostics(
    path: Path,
) -> list[list[list[dict[str, object]]]] | None:
    result = json.loads(path.read_text(encoding="utf-8"))
    runs = result.get("runs", [])
    if not runs or any("output_token_diagnostics" not in run for run in runs):
        return None
    return [run["output_token_diagnostics"] for run in runs]


def first_mismatch_locations(
    expected: list[list[list[int]]],
    actual: list[list[list[int]]],
) -> list[tuple[int, int, int]]:
    locations = []
    for repeat_index, (expected_run, actual_run) in enumerate(zip(expected, actual)):
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
            locations.append((repeat_index, request_index, mismatch_index))
    return locations


def format_token_diagnostic(label: str, diagnostic: dict[str, object]) -> str:
    return (
        f"  {label}: top={diagnostic['top_token_ids']}, "
        f"logits={diagnostic['top_logits']}, margin={diagnostic['margin']}, "
        f"mixed={diagnostic['mixed_step']}, "
        f"prefill={diagnostic['is_prefill']}, "
        f"q/context={diagnostic['query_length']}/"
        f"{diagnostic['context_length']}"
    )


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
            expected_token = (
                expected_tokens[mismatch_index]
                if mismatch_index < len(expected_tokens)
                else "<missing>"
            )
            actual_token = (
                actual_tokens[mismatch_index]
                if mismatch_index < len(actual_tokens)
                else "<missing>"
            )
            mismatches.append(
                f"repeat {repeat_index}, request {request_index}, "
                f"token {mismatch_index}: expected {expected_token}, "
                f"actual {actual_token}"
            )
    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Compare greedy benchmark outputs token by token."
    )
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    expected = load_token_ids(args.expected)
    actual = load_token_ids(args.actual)
    mismatches = compare_token_ids(expected, actual)
    if mismatches:
        print(f"FAIL: {len(mismatches)} request(s) have different greedy token IDs")
        for mismatch in mismatches[:20]:
            print(f"- {mismatch}")
        expected_diagnostics = load_token_diagnostics(args.expected)
        actual_diagnostics = load_token_diagnostics(args.actual)
        if expected_diagnostics is not None and actual_diagnostics is not None:
            print("First-mismatch logit diagnostics:")
            for repeat_index, request_index, token_index in first_mismatch_locations(
                expected,
                actual,
            )[:20]:
                print(
                    f"- repeat {repeat_index}, request {request_index}, "
                    f"token {token_index}"
                )
                print(
                    format_token_diagnostic(
                        "expected",
                        expected_diagnostics[repeat_index][request_index][token_index],
                    )
                )
                print(
                    format_token_diagnostic(
                        "actual",
                        actual_diagnostics[repeat_index][request_index][token_index],
                    )
                )
        raise SystemExit(1)
    print("PASS: all greedy token IDs are identical")


if __name__ == "__main__":
    main()
