import torch
from torch import nn

from ..complex_layers import ComplexBatchNorm1d, ComplexConv1d
from .complex_transformer import ComplexTransformerEncoder


class NMformer(nn.Module):
    """Transformer backbone with noise token augmentation."""

    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        num_noise_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_noise_tokens = num_noise_tokens
        self.frontend = nn.Sequential(
            ComplexConv1d(1, d_model, kernel_size=3, padding=1),
            ComplexBatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.transformer = ComplexTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )
        total_len = seq_len + num_noise_tokens
        self.classifier = nn.Linear(2 * d_model * total_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        noise = torch.randn(
            x.size(0), x.size(1), self.num_noise_tokens, device=x.device
        )
        x = torch.cat([x, noise], dim=2)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.classifier(x)
