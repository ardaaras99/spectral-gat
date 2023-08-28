import torch


def foo():
    n = 250
    n_classes = 7
    # Parameters
    a1 = torch.rand(16, 1)
    a2 = torch.rand(16, 1)
    W = torch.rand(64, 16)

    X = torch.rand(n, 64)

    K = X @ W

    k1 = K @ a1
    k2 = K @ a2
    E = k1 + k2.view(1, n)
    return E, k1, k2, n


def test_foo():
    E, k1, k2, n = foo()
    for i in range(n):
        for j in range(n):
            assert torch.allclose(E[i, j], k1[i] + k2[j], atol=1e-6)
