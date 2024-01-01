from torch import nn
from .attention import MultiheadAttentionBlock
from .ffn import FeedForwardBlock
from .layer_norm import LayerNorm
from .residual import ResidualConnection

# Transformers Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, L: int, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout_rate, norm_first=norm_first) for _ in range(L)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.layer_norm(x)
    

# One decoder block
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h:int, d_ff:int, dropout_rate: float, norm_first: bool):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = MultiheadAttentionBlock(d_model, h, dropout_rate)
        self.cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout_rate)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_rate) for _ in range(3)])
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(3)])
        self.norm_first = norm_first
    
    def _pre_LN(self, x, encoder_output, src_mask, tgt_mask):
        # causal self attention
        self_attn_input = self.layer_norms[0](x)
        self_attn_output = self.self_attention_block(self_attn_input, self_attn_input, self_attn_input, tgt_mask)
        x = self.residual_connections[0](x, self_attn_output)
        # cross attention
        cross_attn_input = self.layer_norms[1](x)
        cross_attn_output = self.cross_attention_block(cross_attn_input, encoder_output, encoder_output, src_mask)
        x = self.residual_connections[1](x, cross_attn_output)
        # feedforward block
        ffn_input = self.layer_norms[2](x)
        ffn_output = self.feed_forward_block(ffn_input)
        x = self.residual_connections[2](x, ffn_output)
        return x

    def _post_LN(self, x, encoder_output, src_mask, tgt_mask):
        # causal self attention
        self_attn_output = self.self_attention_block(x, x, x, tgt_mask)
        x = self.residual_connections[0](x, self_attn_output)
        x = self.layer_norms[0](x)
        # cross attention
        cross_attn_output = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        x = self.residual_connections[1](x, cross_attn_output)
        x = self.layer_norms[1](x)
        # feedforward block
        ffn_output = self.feed_forward_block(x)
        x = self.residual_connections[2](x, ffn_output)
        x = self.layer_norms[2](x)
        return x

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        if self.norm_first:       
            return self._pre_LN(x, encoder_output, src_mask, tgt_mask)
        else:                    
            return self._post_LN(x, encoder_output, src_mask, tgt_mask)
