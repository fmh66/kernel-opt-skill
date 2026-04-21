import torch

atol = 1e-3
rtol = 1e-2


def reference(Q, K, V, output, N, d_model, num_heads):
    """PyTorch reference for multi-head attention.

    Buffers are flat 1D tensors of size >= N*d_model.
    Layout: tensor[i * d_model + head * dk + d]
    """
    dk = d_model // num_heads

    Q_mat = Q[:N * d_model].view(N, d_model)
    K_mat = K[:N * d_model].view(N, d_model)
    V_mat = V[:N * d_model].view(N, d_model)

    # (h, N, dk)
    Q_h = Q_mat.view(N, num_heads, dk).permute(1, 0, 2).contiguous()
    K_h = K_mat.view(N, num_heads, dk).permute(1, 0, 2).contiguous()
    V_h = V_mat.view(N, num_heads, dk).permute(1, 0, 2).contiguous()

    scale = dk ** -0.5
    scores = torch.bmm(Q_h, K_h.transpose(1, 2)) * scale   # (h, N, N)
    attn = torch.softmax(scores, dim=-1)                    # (h, N, N)
    out = torch.bmm(attn, V_h)                              # (h, N, dk)

    # (N, num_heads, dk) → (N, d_model)
    out = out.permute(1, 0, 2).contiguous().view(N, d_model)
    output[:N * d_model].copy_(out.view(-1))
