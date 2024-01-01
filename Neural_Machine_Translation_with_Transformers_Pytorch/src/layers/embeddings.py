import math
import torch
from torch import nn


# Input embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
    


# Positional encodings
class PositionalEmbeddings(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout_rate: float):
        super(PositionalEmbeddings, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)

        # a matrix of shape (seq_len, d_model)
        pe = torch.zeros(size=(seq_len, d_model))
        # create a vector of shape (seq_len, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp((-math.log(10000.0)/d_model)*torch.arange(0, d_model, 2, dtype=torch.float32))
        # if odd feature dimensionality
        end = d_model - div_term.shape[-1]
        # Apply sin to even positions of embeddings
        pe[:, 0::2] = torch.sin(pos*div_term)
        # Apply cos term to odd positions of embeddings
        pe[:, 1::2] = torch.cos(pos*div_term[:end])
        # add an extra dim (for batch_size) to the pe matrix
        pe = torch.unsqueeze(pe, 0) # (1, seq_len, d_model)
        # save the encoding in module buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)