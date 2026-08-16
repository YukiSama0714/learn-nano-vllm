import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_batch_stride,
    rows_per_batch: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // rows_per_batch
    row_in_batch = row % rows_per_batch
    x_row_offset = batch * x_batch_stride + row_in_batch * hidden_size
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    x = tl.load(
        x_ptr + x_row_offset + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(
        output_ptr + row * hidden_size + offsets,
        x * inverse_rms * weight,
        mask=mask,
    )


@triton.jit
def add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_output_ptr,
    x_batch_stride,
    residual_batch_stride,
    rows_per_batch: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // rows_per_batch
    row_in_batch = row % rows_per_batch
    x_row_offset = batch * x_batch_stride + row_in_batch * hidden_size
    residual_row_offset = (
        batch * residual_batch_stride + row_in_batch * hidden_size
    )
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    x = tl.load(
        x_ptr + x_row_offset + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + residual_row_offset + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    combined = x + residual
    tl.store(
        residual_output_ptr + row * hidden_size + offsets,
        combined,
        mask=mask,
    )
    variance = tl.sum(combined * combined, axis=0) / hidden_size
    inverse_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(
        output_ptr + row * hidden_size + offsets,
        combined * inverse_rms * weight,
        mask=mask,
    )


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(variance + eps))
    return x.to(orig_dtype).mul_(weight)


def add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual_output = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(variance + eps))
    output = x.to(orig_dtype).mul_(weight)
    return output, residual_output


def _validate_triton_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> int:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("Triton RMSNorm requires CUDA tensors")
    if x.ndim < 2:
        raise ValueError("Triton RMSNorm requires at least two dimensions")
    if not weight.is_contiguous():
        raise ValueError("Triton RMSNorm requires a contiguous weight")
    expected_stride = 1
    for dimension in range(x.ndim - 1, 0, -1):
        if x.stride(dimension) != expected_stride:
            raise ValueError(
                "Triton RMSNorm requires packed dimensions after dim 0"
            )
        expected_stride *= x.size(dimension)
    hidden_size = x.size(-1)
    if weight.numel() != hidden_size:
        raise ValueError("weight size must match the last input dimension")
    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError("residual and input shapes must match")
        if not residual.is_cuda:
            raise ValueError(
                "Triton RMSNorm requires a CUDA residual"
            )
        expected_stride = 1
        for dimension in range(residual.ndim - 1, 0, -1):
            if residual.stride(dimension) != expected_stride:
                raise ValueError(
                    "Triton RMSNorm requires packed residual dimensions "
                    "after dim 0"
                )
            expected_stride *= residual.size(dimension)
    return hidden_size


def _num_warps(hidden_size: int) -> int:
    if hidden_size <= 256:
        return 1
    if hidden_size <= 2048:
        return 4
    return 8


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    hidden_size = _validate_triton_inputs(x, weight)
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_rows = x.numel() // hidden_size
    rows_per_batch = x[0].numel() // hidden_size
    block_size = triton.next_power_of_2(hidden_size)
    rms_norm_kernel[(num_rows,)](
        x,
        weight,
        output,
        x.stride(0),
        rows_per_batch,
        hidden_size,
        eps,
        block_size,
        num_warps=_num_warps(hidden_size),
    )
    return output


def triton_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = _validate_triton_inputs(x, weight, residual)
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    residual_output = torch.empty_like(
        residual,
        memory_format=torch.contiguous_format,
    )
    num_rows = x.numel() // hidden_size
    rows_per_batch = x[0].numel() // hidden_size
    block_size = triton.next_power_of_2(hidden_size)
    add_rms_norm_kernel[(num_rows,)](
        x,
        residual,
        weight,
        output,
        residual_output,
        x.stride(0),
        residual.stride(0),
        rows_per_batch,
        hidden_size,
        eps,
        block_size,
        num_warps=_num_warps(hidden_size),
    )
    return output, residual_output


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        backend: str = "compiled",
    ) -> None:
        super().__init__()
        if backend not in {"eager", "compiled", "triton"}:
            raise ValueError(f"unsupported RMSNorm backend: {backend}")
        self.eps = eps
        self.backend = backend
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return rms_norm_reference(x, self.weight, self.eps)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return add_rms_norm_reference(x, residual, self.weight, self.eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "triton":
            if residual is None:
                return triton_rms_norm(x, self.weight, self.eps)
            return triton_add_rms_norm(x, residual, self.weight, self.eps)
        if self.backend == "eager":
            if residual is None:
                return rms_norm_reference(x, self.weight, self.eps)
            return add_rms_norm_reference(
                x,
                residual,
                self.weight,
                self.eps,
            )
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
