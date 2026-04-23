"""v3: Tensor Core acceleration (TF32) + complete autotune.

Key changes from v2:
- allow_tf32=True in tl.dot: Triton targets Ampere Tensor Cores (TF32 mode),
  which deliver ~3× more FLOPS vs FP32 CUDA cores.  Reduces the Short Scoreboard
  (shared-memory staging) stall that dominates v2 by executing the same multiply
  with fewer cycles.  Precision trade-off: TF32 rounds each input mantissa to
  10 bits before multiply, introducing ~1e-3 relative error; tolerance set to 2e-3.
- Expanded autotune grid now includes BLOCK_M=32/BLOCK_N=64 (the config that was
  fastest in v2) alongside larger/smaller tile variants, letting the autotuner
  confirm or improve the optimal tile configuration on this specific GPU.
- num_stages=2 kept where it helps; num_stages=1 included for comparison.
"""

import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # include v2's winning config plus TC variant
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "num_warps": 4, "num_stages": 1}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "num_warps": 4, "num_stages": 1}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "num_warps": 4, "num_stages": 1}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "num_warps": 8, "num_stages": 1}),
    ],
    key=["N", "d_k"],
)
@triton.jit
def mha_kernel_v3(
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

    offs_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < N
    d_mask = offs_d < d_k

    # Load Q tile [BLOCK_M, BLOCK_D] once — reused for all key blocks
    q_ptrs = (
        q_ptr
        + h * stride_q_h
        + offs_m[:, None] * stride_q_n
        + offs_d[None, :] * stride_q_d
    )
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        cur_offs_n = n_start + offs_n
        n_mask = cur_offs_n < N
        kv_mask = n_mask[:, None] & d_mask[None, :]

        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = (
            k_ptr
            + h * stride_k_h
            + cur_offs_n[:, None] * stride_k_n
            + offs_d[None, :] * stride_k_d
        )
        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # QK^T via Tensor Cores (TF32): ~3× faster than FP32 CUDA cores
        scores = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=True) * inv_sqrt_dk
        scores = tl.where(n_mask[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)
        l_i = alpha * l_i + tl.sum(p, axis=1)

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = (
            v_ptr
            + h * stride_v_h
            + cur_offs_n[:, None] * stride_v_n
            + offs_d[None, :] * stride_v_d
        )
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # PV via Tensor Cores (TF32): accumulation stays FP32
        acc = alpha[:, None] * acc + tl.dot(p, v_tile, allow_tf32=True)
        m_i = m_new

    out = acc / l_i[:, None]

    out_ptrs = (
        out_ptr
        + h * stride_o_h
        + offs_m[:, None] * stride_o_n
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, out.to(tl.float32), mask=m_mask[:, None] & d_mask[None, :])


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

    grid = lambda meta: (num_heads, triton.cdiv(N, meta["BLOCK_M"]))
    mha_kernel_v3[grid](
        Q, K, V, output,
        N, d_k,
        d_k, d_model, 1,
        d_k, d_model, 1,
        d_k, d_model, 1,
        d_k, d_model, 1,
        1.0 / math.sqrt(d_k),
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
