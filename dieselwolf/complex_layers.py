import torch
from torch import nn


class ComplexConv1d(nn.Module):
    """Complex 1D convolution implemented with separate real and imaginary weights."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.imag = nn.Conv1d(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=1)
        real = self.real(x_r) - self.imag(x_i)
        imag = self.real(x_i) + self.imag(x_r)
        return torch.cat([real, imag], dim=1)


class ComplexBatchNorm1d(nn.Module):
    """BatchNorm for complex inputs operating on real and imaginary parts separately."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.real = nn.BatchNorm1d(num_features)
        self.imag = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=1)
        real = self.real(x_r)
        imag = self.imag(x_i)
        return torch.cat([real, imag], dim=1)


class ComplexLinear(nn.Module):
    """Linear layer for complex inputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=-1)
        real = self.real(x_r) - self.imag(x_i)
        imag = self.real(x_i) + self.imag(x_r)
        return torch.cat([real, imag], dim=-1)
