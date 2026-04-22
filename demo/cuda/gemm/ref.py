import torch


def reference(**kwargs):
    A = kwargs['A']
    B = kwargs['B']
    C = kwargs['C']
    M = kwargs['M']
    K = kwargs['K']
    N = kwargs['N']
    result = torch.mm(A[:M * K].view(M, K), B[:K * N].view(K, N))
    C[:M * N].copy_(result.view(-1))
