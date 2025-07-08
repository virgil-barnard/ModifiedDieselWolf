import math
import torch
from torch import nn

from ..complex_layers import ComplexLinear, ComplexLayerNorm


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal position embeddings for complex inputs."""

    def __init__(self, embed_dim: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.permute(1, 0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, : x.size(2)]


class ComplexTransformerEncoderLayer(nn.Module):
    """Basic complex-valued Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model * 2, nhead, dropout=dropout)
        self.linear1 = ComplexLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = ComplexLinear(dim_feedforward, d_model)

        self.norm1 = ComplexLayerNorm(d_model)
        self.norm2 = ComplexLayerNorm(d_model)
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

        # feed-forward expects features in last dimension
        ff = src.permute(0, 2, 1)
        ff = self.linear1(ff)
        ff = self.activation(ff)
        ff = self.dropout(ff)
        ff = self.linear2(ff)
        ff = ff.permute(0, 2, 1)
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
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ComplexTransformerEncoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        if seq_len is not None:
            self.pos_enc = SinusoidalPositionalEncoding(d_model * 2, seq_len)
        else:
            self.pos_enc = None

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src
        if self.pos_enc is not None:
            out = self.pos_enc(out)
        for layer in self.layers:
            out = layer(out)
        return out
