import argparse
import json
from pathlib import Path

import torch
import triton

from nanovllm.layers.layernorm import RMSNorm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and benchmark eager, torch.compile, and Triton "
            "RMSNorm implementations."
        )
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        nargs="+",
        default=[1, 8, 64, 512, 4096],
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        nargs="+",
        default=[128, 4096],
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=("rms", "add_rms"),
        default=["rms", "add_rms"],
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rows-per-batch", type=int, default=1)
    parser.add_argument("--batch-padding-rows", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value <= 0 for value in args.num_rows + args.hidden_size):
        parser.error("row counts and hidden sizes must be positive")
    if args.eps <= 0 or args.warmup <= 0 or args.rep <= 0:
        parser.error("eps, warmup, and rep must be positive")
    if args.rows_per_batch <= 0 or args.batch_padding_rows < 0:
        parser.error("rows per batch must be positive; padding cannot be negative")
    if any(value % args.rows_per_batch for value in args.num_rows):
        parser.error("each row count must be divisible by --rows-per-batch")
    return args


def make_modules(hidden_size, dtype, eps):
    weight = 1 + 0.1 * torch.randn(
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    modules = {}
    for backend in ("eager", "compiled", "triton"):
        module = RMSNorm(
            hidden_size,
            eps=eps,
            backend=backend,
        ).to(device="cuda", dtype=dtype)
        with torch.no_grad():
            module.weight.copy_(weight)
        modules[backend] = module
    return modules


def run_module(module, operation, x, residual):
    if operation == "rms":
        return module(x)
    return module(x, residual)


def assert_close(actual, expected):
    if isinstance(expected, tuple):
        torch.testing.assert_close(
            actual[0],
            expected[0],
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            actual[1],
            expected[1],
            rtol=0,
            atol=0,
        )
        return
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def max_abs_error(actual, expected):
    if isinstance(expected, tuple):
        return max(
            (actual_tensor.float() - expected_tensor.float())
            .abs()
            .max()
            .item()
            for actual_tensor, expected_tensor in zip(actual, expected)
        )
    return (actual.float() - expected.float()).abs().max().item()


def transferred_bytes(operation, num_rows, hidden_size, element_size):
    tensor_elements = num_rows * hidden_size
    tensor_copies = 2 if operation == "rms" else 4
    return (tensor_copies * tensor_elements + hidden_size) * element_size


def make_inputs(args, num_rows, hidden_size, dtype):
    if args.rows_per_batch == 1 and args.batch_padding_rows == 0:
        x = torch.randn(
            num_rows,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        return x, torch.randn_like(x)
    num_batches = num_rows // args.rows_per_batch
    stored_rows = args.rows_per_batch + args.batch_padding_rows
    x_storage = torch.randn(
        num_batches,
        stored_rows * hidden_size,
        device="cuda",
        dtype=dtype,
    )
    residual_storage = torch.randn_like(x_storage)
    logical_elements = args.rows_per_batch * hidden_size
    logical_shape = (
        num_batches,
        args.rows_per_batch,
        hidden_size,
    )
    x = x_storage[:, :logical_elements].view(logical_shape)
    residual = residual_storage[:, :logical_elements].view(logical_shape)
    return x, residual


def benchmark_case(
    args,
    modules,
    operation,
    num_rows,
    hidden_size,
    dtype,
):
    x, residual = make_inputs(args, num_rows, hidden_size, dtype)
    outputs = {
        backend: run_module(module, operation, x, residual)
        for backend, module in modules.items()
    }
    torch.cuda.synchronize()
    assert_close(outputs["compiled"], outputs["eager"])
    assert_close(outputs["triton"], outputs["eager"])

    timings = {
        backend: triton.testing.do_bench(
            lambda module=module: run_module(
                module,
                operation,
                x,
                residual,
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
        for backend, module in modules.items()
    }
    bytes_moved = transferred_bytes(
        operation,
        num_rows,
        hidden_size,
        x.element_size(),
    )
    return {
        "operation": operation,
        "num_rows": num_rows,
        "hidden_size": hidden_size,
        "eager_ms": round(timings["eager"], 6),
        "compiled_ms": round(timings["compiled"], 6),
        "triton_ms": round(timings["triton"], 6),
        "compiled_max_abs_error": round(
            max_abs_error(outputs["compiled"], outputs["eager"]),
            8,
        ),
        "triton_max_abs_error": round(
            max_abs_error(outputs["triton"], outputs["eager"]),
            8,
        ),
        "triton_vs_eager": round(
            timings["eager"] / timings["triton"],
            3,
        ),
        "triton_vs_compiled": round(
            timings["compiled"] / timings["triton"],
            3,
        ),
        "compiled_effective_gbps": round(
            bytes_moved / timings["compiled"] / 1e6,
            3,
        ),
        "triton_effective_gbps": round(
            bytes_moved / timings["triton"] / 1e6,
            3,
        ),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    results = []
    for hidden_size in args.hidden_size:
        modules = make_modules(hidden_size, dtype, args.eps)
        for operation in args.ops:
            for num_rows in args.num_rows:
                results.append(
                    benchmark_case(
                        args,
                        modules,
                        operation,
                        num_rows,
                        hidden_size,
                        dtype,
                    )
                )
    report = {
        "environment": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": {
            "num_rows": args.num_rows,
            "hidden_size": args.hidden_size,
            "operations": args.ops,
            "dtype": args.dtype,
            "eps": args.eps,
            "warmup": args.warmup,
            "rep": args.rep,
            "seed": args.seed,
            "rows_per_batch": args.rows_per_batch,
            "batch_padding_rows": args.batch_padding_rows,
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
