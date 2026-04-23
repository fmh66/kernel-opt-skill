import torch
import torch.nn.functional as F

# Tolerance widened to 2e-3 to accommodate TF32 rounding in Tensor Core path (v3)
atol = 2e-3
rtol = 2e-3


def reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    num_heads: int,
):
    d_k = d_model // num_heads
    inv_sqrt_dk = d_k ** -0.5

    # [N, num_heads, d_k] -> [num_heads, N, d_k]
    q = Q.view(N, num_heads, d_k).permute(1, 0, 2).contiguous()
    k = K.view(N, num_heads, d_k).permute(1, 0, 2).contiguous()
    v = V.view(N, num_heads, d_k).permute(1, 0, 2).contiguous()

    scores = torch.bmm(q, k.transpose(1, 2)) * inv_sqrt_dk  # [H, N, N]
    attn = F.softmax(scores, dim=-1)                         # [H, N, N]
    context = torch.bmm(attn, v)                             # [H, N, d_k]

    out = context.transpose(0, 1).contiguous().view(-1)
    output.copy_(out)


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
