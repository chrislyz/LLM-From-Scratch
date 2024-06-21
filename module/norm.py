from typing import Tuple

import torch
import torch.nn as nn


class BatchNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-05, momentum: float = 0.1) -> None:
        super().__init__()
        shape = (1, num_features, 1, 1)
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.running_mean = torch.zeros(1)
        self.running_var = torch.ones(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = torch.mean(x, [0, 2, 3], keepdim=True)
            var = torch.var(x, [0, 2, 3], correction=0, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var
        x_normed = (x - mean) * torch.rsqrt(var + self.eps)
        self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var

        # maintain the size for statistics checking
        self.running_mean = self.running_mean.squeeze()
        self.running_var = self.running_var.squeeze()
        return self.gamma * x_normed + self.beta


class LayerNorm(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()  # torch.mean requires computation on floating numbers
        x_rms = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_rms.type_as(x) * self.scale


if __name__ == '__main__':
    def cmp_bn(bn1, bn2):
        err = False
        if not torch.allclose(bn1.running_mean, bn2.running_mean):
            print('Diff in running_mean: {} vs {}'.format(bn1.running_mean, bn2.running_mean))
            err = True
        if not torch.allclose(bn1.running_var, bn2.running_var):
            print('Diff in running_var: {} vs {}'.format(bn1.running_var, bn2.running_var))
            err = True
        if not err:
            print('Test passed!')
    b1 = nn.BatchNorm2d(100)
    b2 = BatchNorm(100)
    x = torch.randn(20, 100, 35, 45)
    y1 = b1(x)
    y2 = b2(x)
    print(y1.size(), y2.size())
    print(b1.running_mean.size(), b2.running_mean.size())
    cmp_bn(b1, b2)
