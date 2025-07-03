import torch
from torch import nn
from typing import Sequence

from ..complex_layers import ComplexBatchNorm1d, ComplexConv1d
from .complex_transformer import ComplexTransformerEncoder


class ConfigurableNMformer(nn.Module):
    """Flexible version of :class:`NMformer` with noise token augmentation."""

    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        conv_channels: Sequence[int] | int = 32,
        kernel_sizes: Sequence[int] | int = 3,
        batch_norm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
        nhead: int = 4,
        num_layers: int = 2,
        num_noise_tokens: int = 2,
    ) -> None:
        super().__init__()
        if isinstance(conv_channels, int):
            conv_channels = [conv_channels]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have same length")

        act_map = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "tanh": nn.Tanh,
        }
        act_cls = act_map.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'")

        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k in zip(conv_channels, kernel_sizes):
            layers.append(ComplexConv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            if batch_norm:
                layers.append(ComplexBatchNorm1d(out_ch))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.frontend = nn.Sequential(*layers)

        self.num_noise_tokens = num_noise_tokens
        total_len = seq_len + num_noise_tokens

        self.transformer = ComplexTransformerEncoder(
            d_model=in_ch,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=total_len,
        )
        self.classifier = nn.Linear(2 * in_ch * total_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        noise = torch.randn(
            x.size(0), x.size(1), self.num_noise_tokens, device=x.device
        )
        x = torch.cat([x, noise], dim=2)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.classifier(x)
