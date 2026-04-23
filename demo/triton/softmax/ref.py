import torch


def reference(input: torch.Tensor, output: torch.Tensor, N: int, D: int):
    result = torch.softmax(input.float(), dim=-1).to(input.dtype)
    output.copy_(result.view(-1))


def setup(N=1024, D=1024, seed=42, dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    input_tensor = torch.randn((N, D), device="cuda", dtype=dtype)
    output_tensor = torch.empty((N, D), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "input": input_tensor,
            "output": output_tensor,
            "N": int(N),
            "D": int(D),
        },
        "outputs": ["output"],
    }
