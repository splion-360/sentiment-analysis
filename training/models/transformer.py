import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT, EMBEDDING_DIM, FF_PROJECTION_DIM, MAX_LENGTH, NUM_CLASSES, NUM_HEADS


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.pos_encoding = PositionalEncoding(EMBEDDING_DIM, MAX_LENGTH)
        self.transformer = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_PROJECTION_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        x = self.transformer(x, attention_mask)

        final_token = x[:, -1, :]
        return self.classifier(final_token)
