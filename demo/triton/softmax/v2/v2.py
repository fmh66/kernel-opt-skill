import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4}),
        triton.Config({"num_warps": 8}),
    ],
    key=["D"],
)
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    D,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles 2 consecutive rows to pipeline DRAM latency:
    # row1's load is issued before row0's computation, so DRAM latency
    # for row1 is hidden behind row0's reduction and exp passes.
    block = tl.program_id(0)
    row0 = block * 2
    row1 = block * 2 + 1

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    # Issue both loads upfront before any compute
    row0_data = tl.load(
        input_ptr + row0 * stride_input_row + col_offsets, mask=mask, other=-float("inf")
    )
    # row1 may be out of range when N is odd; use row0 as a dummy pointer
    row1_valid = row1 < N
    row1_ptr_row = tl.where(row1_valid, row1, row0)
    row1_data = tl.load(
        input_ptr + row1_ptr_row * stride_input_row + col_offsets,
        mask=mask,
        other=-float("inf"),
    )

    # Process row0 (DRAM latency for row1 is hidden here)
    max0 = tl.max(row0_data, axis=0)
    num0 = tl.exp(row0_data - max0)
    sum0 = tl.sum(num0, axis=0)
    tl.store(
        output_ptr + row0 * stride_output_row + col_offsets, num0 / sum0, mask=mask
    )

    # Process row1
    max1 = tl.max(row1_data, axis=0)
    num1 = tl.exp(row1_data - max1)
    sum1 = tl.sum(num1, axis=0)
    if row1_valid:
        tl.store(
            output_ptr + row1 * stride_output_row + col_offsets,
            num1 / sum1,
            mask=mask,
        )


def solve(input: torch.Tensor, output: torch.Tensor, N: int, D: int):
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = ((N + 1) // 2,)
    softmax_kernel[grid](
        input,
        output,
        N,
        D,
        input.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def setup(N=1024, D=1024, seed=42, dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    input_tensor = torch.randn((N, D), device="cuda", dtype=dtype)
    output_tensor = torch.empty((N, D), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "input": input_tensor,
            "output": output_tensor,
            "N": int(N),
            "D": int(D),
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["input"],
        kwargs["output"],
        int(kwargs["N"]),
        int(kwargs["D"]),
    )
