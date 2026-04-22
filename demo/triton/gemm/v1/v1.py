import torch
import triton
import triton.language as tl


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
    num_stages: tl.constexpr,
):
    # Grouped ordering: improves L2 cache reuse by grouping pid_m blocks
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=num_stages):
        k_remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < k_remaining,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < k_remaining,
            other=0.0,
        )
        # tf32: enables Tensor Core MMA on Ampere (sm_86), replaces ieee FMA
        acc = tl.dot(a, b, acc, input_precision="tf32")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float32)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, K: int, N: int):
    M = int(M)
    K = int(K)
    N = int(N)

    if A.device.type != "cuda" or B.device.type != "cuda" or C.device.type != "cuda":
        raise ValueError("A, B, C must be CUDA tensors.")

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    num_stages = 3
    num_warps = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
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
