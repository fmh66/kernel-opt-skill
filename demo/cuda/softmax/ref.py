import torch
import torch.nn.functional as F


def reference(input: torch.Tensor, output: torch.Tensor, N: int, D: int) -> torch.Tensor:
    x = input.view(N, D)
    result = F.softmax(x, dim=-1)
    output.copy_(result.view(-1))
    return output
