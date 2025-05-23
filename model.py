import torch
import torch.nn as nn
import math
import numpy as np

class WordEmbeddings(nn.Module):
    """
        Creates a unique embedding for each word in the vocabulary
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
        Create a unique embedding for each position up to a maximum sequence length 'seq_len'.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)# (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        denom = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -math.log(10000.0) / d_model) # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(postion * denom)

        pe = pe.unequeeze(0) # (1, seq_len, d_model)
        # The tensor will be saved along with the learned parameters when the model is saved
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 'x' will have dimension (Batch, seq_len, d_model)
        # Adds the positional encoding up to the sentence size
        x += self.pe[:, :x.shape[1], :].requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
        Normalizes the last dimension of input while scaling with learned parameters
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__():
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean)/(self.std + eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
        A fully connected layer with ReLU activation and dropout followed by another fully connected layer
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
        Runs the input through a multi-head attention operation
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        # Dividing embedding to 'h' heads to operate on smaller parts
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.d_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, head, seq_len, d_k) --> (Batch, head, seq_len, seq_len)
        attention_scores = (query @ key.tanspose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.mask_fill_(mask == 0, -1e9)
            attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, seq_len, d_model) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
        Adds the tensor to the output of previous layer
    """
    def __init__(self, dropout: float) -> None:
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
        Combines all previously created blocks to make Encoder
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """
    Applying multuple iterations of the EncoderBlock
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
        Combines all previously created blocks and output of EncoderBlock to make Decoder
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """
        Apply multiple iterations of the DecoderBlock
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """

    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.lof_softmax(self.proj(x), dim = -1)
