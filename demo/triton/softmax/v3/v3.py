import torch
import triton
import triton.language as tl


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

    # max_contiguous hint: tells the compiler the offsets are fully sequential,
    # enabling 128-bit vectorized LDG/STG instructions.
    col_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask = col_offsets < D

    row_data = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))

    max_val = tl.max(row_data, axis=0)
    numerator = tl.exp(row_data - max_val)
    sum_val = tl.sum(numerator, axis=0)
    softmax_out = numerator / sum_val

    tl.store(row_output_ptr + col_offsets, softmax_out, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, D: int):
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)
    # num_warps=2: 64 threads/block
    #   - 2-warp inter-warp reduction uses fewer shared mem writes + barriers
    #     than 4-warp reduction (v0), reducing Short Scoreboard + Barrier stalls.
    #   - Register budget allows 24 concurrent blocks/SM → 48 warps → 100% occupancy.
    softmax_kernel[grid](
        input,
        output,
        N,
        D,
        input.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=2,
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
