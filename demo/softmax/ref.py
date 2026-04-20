import torch


def reference(input: torch.Tensor, output: torch.Tensor, N: int, D: int) -> None:
    x = input.view(N, D)
    result = torch.softmax(x.float(), dim=-1)
    output.copy_(result.flatten())
