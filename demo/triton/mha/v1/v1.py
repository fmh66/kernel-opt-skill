"""v1: grid (H, N) — one program per query token per head.

Key changes from v0:
- Grid changed from (H, N, d_k) to (H, N): eliminates d_k-fold redundant softmax computation
- Each program computes the full output row output[h, i, :d_k] in one pass
- V is now accessed row-wise [BLOCK_N, BLOCK_D] tiles instead of column-wise scalars
- Accumulator is a BLOCK_D-length vector, enabling coalesced output store
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def mha_kernel_v1(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
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
    h = tl.program_id(0)
    i = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_k

    # Load q[h, i, :] — one full head vector per program
    q_ptrs = q_ptr + h * stride_q_h + i * stride_q_n + offs_d * stride_q_d
    q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)  # [BLOCK_D]

    # Online softmax accumulators
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = -float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Load K tile [BLOCK_N, BLOCK_D]: K[h, n_start:n_start+BLOCK_N, :]
        k_ptrs = (
            k_ptr
            + h * stride_k_h
            + offs_n[:, None] * stride_k_n
            + offs_d[None, :] * stride_k_d
        )
        k_mask = n_mask[:, None] & d_mask[None, :]
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Scores [BLOCK_N]: dot(K_tile, q_vec)
        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * inv_sqrt_dk
        scores = tl.where(n_mask, scores, -float("inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)  # [BLOCK_N]
        l_i = alpha * l_i + tl.sum(p, axis=0, keep_dims=True)

        # Load V tile [BLOCK_N, BLOCK_D]: V[h, n_start:n_start+BLOCK_N, :]
        v_ptrs = (
            v_ptr
            + h * stride_v_h
            + offs_n[:, None] * stride_v_n
            + offs_d[None, :] * stride_v_d
        )
        v_tile = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Accumulate weighted V: sum_n(p[n] * V[n, :]) into [BLOCK_D]
        acc = alpha * acc + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_new

    # Normalize and store
    out = acc / l_i
    out_ptrs = out_ptr + h * stride_o_h + i * stride_o_n + offs_d * stride_o_d
    tl.store(out_ptrs, out.to(tl.float32), mask=d_mask)


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
    block_d = triton.next_power_of_2(d_k)
    block_n = 64

    # Q/K/V/output are all [N, d_model]; treat as [H, N, d_k] via strides:
    # stride_h = d_k (interleaved heads), stride_n = d_model, stride_d = 1
    grid = (num_heads, N)
    mha_kernel_v1[grid](
        Q, K, V, output,
        N, d_k,
        d_k, d_model, 1,   # Q strides (h, n, d)
        d_k, d_model, 1,   # K strides
        d_k, d_model, 1,   # V strides
        d_k, d_model, 1,   # output strides
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
