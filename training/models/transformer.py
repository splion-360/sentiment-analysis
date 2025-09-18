import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MAX_LENGTH, TransformerConfig


class PositionalEncoding(nn.Module):

    def __init__(self, d_embed: int, max_len: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        L = x.shape[1]
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_embed, num_heads):
        super().__init__()
        self.d_embed = d_embed
        self.num_heads = num_heads
        self.head_dim = d_embed // num_heads

        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.out_linear = nn.Linear(d_embed, d_embed)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        q = self.q_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)

        output = output.transpose(1, 2).contiguous().view(B, L, self.d_embed)
        return self.out_linear(output)


class FeedForward(nn.Module):

    def __init__(self, d_embed: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_ff)
        self.linear2 = nn.Linear(d_ff, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, d_embed: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_embed, num_heads)
        self.ln1 = nn.LayerNorm(d_embed)
        self.ln2 = nn.LayerNorm(d_embed)
        self.ff = FeedForward(d_embed, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.ln2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):

    def __init__(self, vocab_size: int, padding_idx: int):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, TransformerConfig.EMBEDDING_DIM, padding_idx=padding_idx
        )

        nn.init.normal_(self.embedding.weight)

        self.pos_encoding = PositionalEncoding(
            TransformerConfig.EMBEDDING_DIM, MAX_LENGTH, TransformerConfig.DROPOUT
        )
        self.transformer = nn.ModuleList()

        for _ in range(TransformerConfig.LAYERS):
            self.transformer.append(
                TransformerBlock(
                    TransformerConfig.EMBEDDING_DIM,
                    TransformerConfig.NUM_HEADS,
                    TransformerConfig.FF_PROJECTION_DIM,
                    TransformerConfig.DROPOUT,
                )
            )

        self.dropout = nn.Dropout(TransformerConfig.DROPOUT)
        self.classifier = nn.Linear(TransformerConfig.EMBEDDING_DIM, TransformerConfig.NUM_CLASSES)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        for i, _transformer in enumerate(self.transformer):
            x = _transformer(x, mask[:, None, None, :])

        final_token = x[:, 0, :]
        return self.classifier(final_token)
