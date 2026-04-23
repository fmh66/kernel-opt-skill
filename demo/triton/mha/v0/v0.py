"""v0: baseline — one program per output scalar output[h, i, d].

Bug fix from original: tensor v was passed instead of v.stride(0) for stride_v_h.
Structural fix: use original [N, d_model] tensor layout with explicit strides to
avoid extra elementwise CUDA kernels from permute/contiguous/copy_ that break
single-kernel profiling.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    H,
    N,
    d_k,
    stride_q_h,
    stride_q_n,
    stride_q_d,
    stride_k_h,
    stride_k_n,
    stride_k_d,
    stride_v_h,
    stride_v_n,
    stride_v_d,
    stride_o_h,
    stride_o_n,
    stride_o_d,
    inv_sqrt_dk,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program computes one output element: output[h, i, d]
    h = tl.program_id(0)
    i = tl.program_id(1)
    d = tl.program_id(2)

    if h >= H or i >= N or d >= d_k:
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_k

    q_ptrs = q_ptr + h * stride_q_h + i * stride_q_n + offs_d * stride_q_d
    q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = 0.0

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k_ptrs = (
            k_ptr
            + h * stride_k_h
            + offs_n[:, None] * stride_k_n
            + offs_d[None, :] * stride_k_d
        )
        k_mask = n_mask[:, None] & d_mask[None, :]
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * inv_sqrt_dk
        scores = tl.where(n_mask, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=0)

        v_ptrs = v_ptr + h * stride_v_h + offs_n * stride_v_n + d * stride_v_d
        v_vals = tl.load(v_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        acc = alpha * acc + tl.sum(p * v_vals, axis=0)
        m_i = m_new
        l_i = l_new

    out_val = acc / l_i
    out_loc = out_ptr + h * stride_o_h + i * stride_o_n + d * stride_o_d
    tl.store(out_loc, out_val)


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    num_heads: int,
):
    d_k = d_model // num_heads

    # Use Q/K/V with their original [N, d_model] layout and pass strides so that
    # accessing head h, token i, dim j maps to tensor[i, h*d_k + j].
    # stride_h = d_k (between heads), stride_n = d_model (between tokens), stride_d = 1
    block_d = min(128, triton.next_power_of_2(d_k))
    block_n = 128

    grid = (num_heads, N, d_k)
    fused_mha_kernel[grid](
        Q, K, V, output,
        num_heads, N, d_k,
        # Q strides: view [N, d_model] as [H, N, d_k]
        d_k, d_model, 1,
        # K strides
        d_k, d_model, 1,
        # V strides
        d_k, d_model, 1,
        # output strides (same logical layout)
        d_k, d_model, 1,
        1.0 / math.sqrt(d_k),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output


def setup(N=1024, d_model=1024, num_heads=16, seed=42, dtype=torch.float32, **kwargs):
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    torch.manual_seed(seed)
    Q = torch.randn((N, d_model), device="cuda", dtype=dtype)
    K = torch.randn((N, d_model), device="cuda", dtype=dtype)
    V = torch.randn((N, d_model), device="cuda", dtype=dtype)
    output = torch.empty((N, d_model), device="cuda", dtype=dtype)

    return {
        "inputs": {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "N": int(N),
            "d_model": int(d_model),
            "num_heads": int(num_heads),
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["Q"],
        kwargs["K"],
        kwargs["V"],
        kwargs["output"],
        int(kwargs["N"]),
        int(kwargs["d_model"]),
        int(kwargs["num_heads"]),
    )
