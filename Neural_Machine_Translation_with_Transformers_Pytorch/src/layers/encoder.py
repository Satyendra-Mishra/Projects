from torch import nn
from .attention import MultiheadAttentionBlock
from .ffn import FeedForwardBlock
from .layer_norm import LayerNorm
from .residual import ResidualConnection


# Transformers Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, L: int, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout_rate, norm_first=norm_first) for _ in range(L)])

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


# One encoder block
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = MultiheadAttentionBlock(d_model, h, dropout_rate)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(2)])
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(2)])
        self.norm_first = norm_first

    def _pre_LN(self, x, mask):
        # self attention
        attn_input = self.layer_norms[0](x)
        attn_output = self.self_attention_block(attn_input, attn_input, attn_input, mask)
        x = self.residual_connections[0](x, attn_output)
        # feedforward block
        ffn_input = self.layer_norms[1](x)
        ffn_output = self.feed_forward_block(ffn_input)
        x = self.residual_connections[1](x, ffn_output)
        return x

    def _post_LN(self, x, mask):
        attn_output = self.self_attention_block(x, x, x, mask)
        x = self.residual_connections[0](x, attn_output)
        x = self.layer_norms[0](x)
        # feedforward block
        ffn_output = self.feed_forward_block(x)
        x = self.residual_connections[1](x, ffn_output)
        x = self.layer_norms[1](x)
        return x  
  
    def forward(self, x, mask):
        if self.norm_first:     
            return self._pre_LN(x, mask)   # pre_LN transformer
        else:
            return self._post_LN(x, mask)  # post_LN transformer