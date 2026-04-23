import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_combine(a_max, a_sum, b_max, b_sum):
    """Associative (max, sum_exp) merge for online softmax tree-reduction.

    By using tl.reduce with this combiner, the two separate tl.max + tl.sum
    passes (4 inter-warp barriers) are replaced with a single pass (2 barriers).

    tl.where guards against exp(-inf - (-inf)) = NaN for masked elements:
    the False branch (0.0) is selected by SELP without propagating NaN.
    """
    new_max = tl.maximum(a_max, b_max)
    new_sum = (
        tl.where(a_max > -float("inf"), a_sum * tl.exp(a_max - new_max), 0.0)
        + tl.where(b_max > -float("inf"), b_sum * tl.exp(b_max - new_max), 0.0)
    )
    return new_max, new_sum


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

    col_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask = col_offsets < D

    row_data = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Online softmax single-pass reduction:
    #   initial state (m=x[i], d=1): exp(x[i]-x[i])=1, so sum starts at 1.
    #   masked elements use d=0 to contribute nothing to the final sum.
    m_init = row_data
    d_init = tl.where(
        mask,
        tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32),
        tl.zeros([BLOCK_SIZE], dtype=tl.float32),
    )

    global_max, global_sum = tl.reduce(
        (m_init, d_init), axis=0, combine_fn=_online_softmax_combine
    )

    softmax_out = tl.exp(row_data - global_max) / global_sum

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
        num_warps=4,
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
