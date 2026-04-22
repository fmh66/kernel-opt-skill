import torch


def reference(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, K: int, N: int):
    result = torch.mm(A, B)
    C.copy_(result.view(-1))


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
