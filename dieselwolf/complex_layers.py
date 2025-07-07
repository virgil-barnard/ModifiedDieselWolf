import torch
from torch import nn


class ComplexConv1d(nn.Module):
    """Complex 1D convolution implemented with separate real and imaginary weights."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.imag = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=1)
        real = self.real(x_r) - self.imag(x_i)
        imag = self.real(x_i) + self.imag(x_r)
        return torch.cat([real, imag], dim=1)

    def reset_parameters(self) -> None:
        self.real.reset_parameters()
        self.imag.reset_parameters()


class ComplexBatchNorm1d(nn.Module):
    """BatchNorm for complex inputs operating on real and imaginary parts separately."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.real = nn.BatchNorm1d(num_features)
        self.imag = nn.BatchNorm1d(num_features)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=1)
        real = self.real(x_r)
        imag = self.imag(x_i)
        return torch.cat([real, imag], dim=1)

    def reset_parameters(self) -> None:
        self.real.reset_parameters()
        self.imag.reset_parameters()


class ComplexLinear(nn.Module):
    """Linear layer for complex inputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=-1)
        real = self.real(x_r) - self.imag(x_i)
        imag = self.real(x_i) + self.imag(x_r)
        return torch.cat([real, imag], dim=-1)

    def reset_parameters(self) -> None:
        self.real.reset_parameters()
        self.imag.reset_parameters()


class ComplexLayerNorm(nn.Module):
    """LayerNorm for complex inputs applied separately to real and imaginary parts."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.real = nn.LayerNorm(normalized_shape)
        self.imag = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_r, x_i = x.chunk(2, dim=1)
        real = self.real(x_r.permute(0, 2, 1)).permute(0, 2, 1)
        imag = self.imag(x_i.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.cat([real, imag], dim=1)
