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
    # Each program instance handles one row
    row = tl.program_id(0)
    if row >= N:
        return

    # Pointer to the start of the current row
    row_input_ptr = input_ptr + row * stride_input_row
    row_output_ptr = output_ptr + row * stride_output_row

    # Column offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    # Load the row (masked, out-of-bounds get -inf)
    row_data = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Step 1: Compute max for numerical stability
    max_val = tl.max(row_data, axis=0)

    # Step 2: Compute exp(x - max)
    numerator = tl.exp(row_data - max_val)

    # Step 3: Compute sum of exponentials
    # (masked-out lanes had -inf, so exp(-inf - max) = 0, contributing nothing)
    sum_val = tl.sum(numerator, axis=0)

    # Step 4: Normalize
    softmax_out = numerator / sum_val

    # Store the result
    tl.store(row_output_ptr + col_offsets, softmax_out, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, D: int):
    # BLOCK_SIZE must be a power of 2 >= D for the single-block-per-row approach
    BLOCK_SIZE = triton.next_power_of_2(D)

    # Launch one program per row
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