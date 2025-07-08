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
        pooling: str | None = "max",
        pool_kernel: int = 2,
        use_mag_phase: bool = False,
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

        pool_map: dict[str, nn.Module] = {
            "max": nn.MaxPool1d,
            "avg": nn.AvgPool1d,
            "lp": nn.LPPool1d,
        }

        layers: list[nn.Module] = []
        in_ch = 2
        curr_len = seq_len
        for out_ch, k in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(act_cls())
            if pooling is not None:
                if pooling == "adaptive":
                    out_size = max(1, curr_len // pool_kernel)
                    layers.append(nn.AdaptiveAvgPool1d(out_size))
                    curr_len = out_size
                elif pooling == "lp":
                    layers.append(
                        nn.LPPool1d(2, kernel_size=pool_kernel, stride=pool_kernel)
                    )
                    curr_len //= pool_kernel
                else:
                    pool_cls = pool_map.get(pooling)
                    if pool_cls is None:
                        raise ValueError(f"Unknown pooling '{pooling}'")
                    layers.append(pool_cls(kernel_size=pool_kernel, stride=pool_kernel))
                    curr_len //= pool_kernel
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch * curr_len, num_classes)
        self.use_mag_phase = use_mag_phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.use_mag_phase:
            i, q = x.chunk(2, dim=1)
            mag = torch.sqrt(i**2 + q**2 + 1e-8)
            phs = torch.atan2(q, i)
            x = torch.cat([mag, phs], dim=1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.classifier(x)
