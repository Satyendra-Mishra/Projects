import math
import torch
from torch import nn
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

# for avoiding graph breaking when using torch.compile()
allow_ops_in_compiled_graph()

# Multihead attention block
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature=1.0, attn_dropout_rate=None):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.attn_dropout = nn.Dropout(p=attn_dropout_rate) if attn_dropout_rate else None
    
    def forward(self, queries, keys, values, mask=None):
        # (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.einsum('bhik,bhjk->bhij', queries, keys)/self.temperature
        # apply the mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0, 1e-9)
        # apply the softmax
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        # dropout
        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights)
        # return the results
        return torch.einsum('bhij,bhjk->bhik', attention_weights, values), attention_weights


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout_rate: float):
        super(MultiheadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, "d_model must be divisible by num of heads"
        self.d_k = self.d_model // self.h
        # projection matrices for Q, K, V and attention_weights*V
        self.WQ = nn.Linear(d_model, d_model) # queries
        self.WK = nn.Linear(d_model, d_model) # keys
        self.WV = nn.Linear(d_model, d_model) # values
        self.WO = nn.Linear(d_model, d_model) # final output
        self.scaled_dot_product_attn = ScaledDotProductAttention(temperature=math.sqrt(self.d_k), 
                                                                 attn_dropout_rate=dropout_rate)


    def forward(self, Q, K, V, mask):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        Q_proj = rearrange(self.WQ(Q), 'b t (h dk) -> b h t dk', h=self.h)
        K_proj = rearrange(self.WK(K), 'b t (h dk) -> b h t dk', h=self.h)
        V_proj = rearrange(self.WV(V), 'b t (h dv) -> b h t dv', h=self.h)
        # compute the self attention
        x, self.attention_weights = self.scaled_dot_product_attn(Q_proj, K_proj, V_proj, mask)
        # merge the output from each head: (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, d_model)
        x = rearrange(x, 'b h t dv -> b t (h dv)')
        return self.WO(x)
    

