import torch
from torch import nn

from ..complex_layers import ComplexLinear, ComplexBatchNorm1d


class ComplexTransformerEncoderLayer(nn.Module):
    """Basic complex-valued Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout)
        self.linear1 = ComplexLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = ComplexLinear(dim_feedforward, d_model)

        self.norm1 = ComplexBatchNorm1d(d_model)
        self.norm2 = ComplexBatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: (batch, 2*d_model, seq_len)
        src2 = src.permute(2, 0, 1)  # (seq_len, batch, 2*d_model)
        attn_output, _ = self.self_attn(src2, src2, src2)
        attn_output = attn_output.permute(1, 2, 0)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src


class ComplexTransformerEncoder(nn.Module):
    """Stack of complex Transformer encoder layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = ComplexTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src
        for layer in self.layers:
            out = layer(out)
        return out
