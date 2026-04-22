import torch
import triton
import triton.language as tl


# All configs use tl.range() so num_stages activates real software pipelining.
# Shared memory per launch = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4 * num_stages <= 101376 bytes
@triton.autotune(
    configs=[
        # BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 → 32768 bytes/stage
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),  # v1 reference
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=2, num_warps=8),
        # BLOCK_M=128, BLOCK_N=64 or 64×128 → 24576 bytes/stage (can do 4 stages)
        triton.Config({"BLOCK_M": 128, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M":  64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        # BLOCK_K=64 variants → larger arithmetic intensity, fewer loop iterations
        triton.Config({"BLOCK_M":  64, "BLOCK_N":  64, "BLOCK_K": 64}, num_stages=3, num_warps=4),  # 32768/stage
        triton.Config({"BLOCK_M": 128, "BLOCK_N":  64, "BLOCK_K": 64}, num_stages=2, num_warps=4),  # 49152/stage
        # High-stage small-K to maximally overlap memory and compute
        triton.Config({"BLOCK_M":  64, "BLOCK_N":  64, "BLOCK_K": 32}, num_stages=5, num_warps=4),  # 16384/stage
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_stages=5, num_warps=8),  # 16384/stage
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Unclamped offsets + explicit masks: correct for all M/N/K including non-aligned sizes
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # tl.range (no num_stages arg): autotune's num_stages controls the pipeline depth
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="tf32")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)
    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, K: int, N: int):
    M = int(M)
    K = int(K)
    N = int(N)

    if A.device.type != "cuda" or B.device.type != "cuda" or C.device.type != "cuda":
        raise ValueError("A, B, C must be CUDA tensors.")

    GROUP_SIZE_M = 8
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        GROUP_SIZE_M=GROUP_SIZE_M,
    )


def setup(M=1024, K=1024, N=1024, seed=42, dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    A = torch.randn((M, K), device="cuda", dtype=dtype)
    B = torch.randn((K, N), device="cuda", dtype=dtype)
    C = torch.empty((M, N), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "A": A,
            "B": B,
            "C": C,
            "M": int(M),
            "K": int(K),
            "N": int(N),
        },
        "outputs": ["C"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["A"],
        kwargs["B"],
        kwargs["C"],
        int(kwargs["M"]),
        int(kwargs["K"]),
        int(kwargs["N"]),
    )
