import torch
import triton
import triton.language as tl

# log2(e) for exp2-based softmax: exp(x) = exp2(x * log2(e))
# EX2 hardware instruction is faster than EXP on Ampere
LOG2E = tl.constexpr(1.4426950408889634)


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4}),
        triton.Config({"num_warps": 8}),
        triton.Config({"num_warps": 16}),
        triton.Config({"num_warps": 32}),
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
    row = tl.program_id(0)

    row_input_ptr = input_ptr + row * stride_input_row
    row_output_ptr = output_ptr + row * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    row_data = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))

    max_val = tl.max(row_data, axis=0)

    # exp2 uses hardware EX2 instruction; numerically: exp(x-m) = exp2((x-m)*log2(e))
    # masked-out lanes had -inf → (−inf − m)*LOG2E = −inf → exp2(−inf) = 0 ✓
    numerator = tl.exp2((row_data - max_val) * LOG2E)

    sum_val = tl.sum(numerator, axis=0)
    softmax_out = numerator / sum_val

    tl.store(row_output_ptr + col_offsets, softmax_out, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, D: int):
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)
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
