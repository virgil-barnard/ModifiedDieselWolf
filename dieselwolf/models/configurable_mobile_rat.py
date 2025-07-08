import torch
from torch import nn
from typing import Sequence

from ..complex_layers import ComplexBatchNorm1d, ComplexConv1d
from .complex_transformer import ComplexTransformerEncoder


class ConfigurableMobileRaT(nn.Module):
    """Flexible version of :class:`MobileRaT`."""

    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        conv_channels: Sequence[int] | int = 32,
        kernel_sizes: Sequence[int] | int = 3,
        batch_norm: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
        nhead: int = 4,
        num_layers: int = 2,
        pooling: str | None = "max",
        pool_kernel: int = 2,
        use_mag_phase: bool = True,
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
        in_ch = 1
        curr_len = seq_len
        for out_ch, k in zip(conv_channels, kernel_sizes):
            layers.append(ComplexConv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            if batch_norm:
                layers.append(ComplexBatchNorm1d(out_ch))
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
        self.frontend = nn.Sequential(*layers)

        self.transformer = ComplexTransformerEncoder(
            d_model=in_ch,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=curr_len,
        )

        self.use_mag_phase = use_mag_phase

        self.classifier = nn.Linear(2 * in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mag_phase:
            i_comp, q_comp = x.chunk(2, dim=1)
            mag = torch.sqrt(i_comp**2 + q_comp**2 + 1e-8)
            phs = torch.atan2(q_comp, i_comp)
            x = torch.cat([mag, phs], dim=1)
        x = self.frontend(x)
        x = self.transformer(x)
        x = x.mean(dim=2)
        return self.classifier(x)
