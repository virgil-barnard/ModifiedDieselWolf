import torch
from torch import nn
from typing import Sequence


class ConfigurableCNN(nn.Module):
    """Flexible 1D CNN for AMR experiments."""

    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        conv_channels: Sequence[int] | int = (32,),
        kernel_sizes: Sequence[int] | int = 3,
        batch_norm: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
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
        in_ch = 2
        for out_ch, k in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch * seq_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv(x)
        x = x.flatten(1)
        return self.classifier(x)
