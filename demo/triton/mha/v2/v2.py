"""v2: Flash Attention tiling with tl.dot — BLOCK_M queries per program.

Key changes from v1:
- Grid changed from (H, N) to (H, ceil_div(N, BLOCK_M)): each program handles
  BLOCK_M=32 query tokens simultaneously, amortizing K/V loads over BLOCK_M queries.
- Use tl.dot for both QK^T [BLOCK_M, BLOCK_N] and PV [BLOCK_M, BLOCK_D]
  computations, activating Tensor Cores (TF32 on Ampere sm_86).
- Proper Flash Attention online softmax with per-row m_i/l_i vectors.
- Coalesced loads: Q_tile [BLOCK_M, BLOCK_D] loaded once and reused for all N/BLOCK_N
  iterations; K and V accessed as [BLOCK_N, BLOCK_D] row tiles.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def mha_kernel_v2(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    m_blk = tl.program_id(1)

    offs_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)  # query token indices
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < N
    d_mask = offs_d < d_k

    # Load Q tile [BLOCK_M, BLOCK_D] — loaded once, reused for all key blocks
    q_ptrs = (
        q_ptr
        + h * stride_q_h
        + offs_m[:, None] * stride_q_n
        + offs_d[None, :] * stride_q_d
    )
    q_mask = m_mask[:, None] & d_mask[None, :]
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_D]

    # Online softmax state per query row
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        cur_offs_n = n_start + offs_n
        n_mask = cur_offs_n < N

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = (
            k_ptr
            + h * stride_k_h
            + cur_offs_n[:, None] * stride_k_n
            + offs_d[None, :] * stride_k_d
        )
        kn_mask = n_mask[:, None] & d_mask[None, :]
        k_tile = tl.load(k_ptrs, mask=kn_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_D]

        # Scores [BLOCK_M, BLOCK_N] = Q_tile @ K_tile^T
        scores = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=False) * inv_sqrt_dk

        # Mask out-of-bounds keys
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        # Online softmax: per-row max/sum update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))    # [BLOCK_M]
        alpha = tl.exp(m_i - m_new)                         # [BLOCK_M]
        p = tl.exp(scores - m_new[:, None])                  # [BLOCK_M, BLOCK_N]
        p = tl.where(n_mask[None, :], p, 0.0)
        l_i = alpha * l_i + tl.sum(p, axis=1)               # [BLOCK_M]

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = (
            v_ptr
            + h * stride_v_h
            + cur_offs_n[:, None] * stride_v_n
            + offs_d[None, :] * stride_v_d
        )
        v_tile = tl.load(v_ptrs, mask=kn_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_D]

        # Accumulate: P @ V [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] = [BLOCK_M, BLOCK_D]
        acc = alpha[:, None] * acc + tl.dot(p, v_tile, allow_tf32=False)
        m_i = m_new

    # Normalize rows
    out = acc / l_i[:, None]  # [BLOCK_M, BLOCK_D]

    # Store output[h, m_start:m_start+BLOCK_M, :]
    out_ptrs = (
        out_ptr
        + h * stride_o_h
        + offs_m[:, None] * stride_o_n
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, out.to(tl.float32), mask=q_mask)


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
    block_m = 32
    block_n = 64

    grid = (num_heads, triton.cdiv(N, block_m))
    mha_kernel_v2[grid](
        Q, K, V, output,
        N, d_k,
        d_k, d_model, 1,   # Q strides (h, n, d)
        d_k, d_model, 1,   # K strides
        d_k, d_model, 1,   # V strides
        d_k, d_model, 1,   # output strides
        1.0 / math.sqrt(d_k),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
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
