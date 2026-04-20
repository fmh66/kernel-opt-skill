import torch
import torch.nn.functional as F

def reference(input, output, N, D, **kwargs):
    result = F.softmax(input.reshape(N, D).float(), dim=-1).reshape(-1)
    output.copy_(result)
