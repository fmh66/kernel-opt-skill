import torch


def reference(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
              M: int, K: int, N: int) -> None:
    a = A[:M * K].view(M, K).float()
    b = B[:K * N].view(K, N).float()
    result = torch.mm(a, b)
    C[:M * N].copy_(result.flatten())
