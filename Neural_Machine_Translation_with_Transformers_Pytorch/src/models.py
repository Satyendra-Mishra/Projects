from torch import nn
from .layers.encoder import TransformerEncoder
from .layers.decoder import TransformerDecoder
from .layers.embeddings import InputEmbeddings, PositionalEmbeddings


def build_transformer(src_vocab_size, tgt_vocab_size, seq_len=128, d_model=512, 
                      h=8, L_enc=6, L_dec=6, d_ff=2048, norm_first=False):
    # source embeddings
    src_input_embd = InputEmbeddings(src_vocab_size, d_model)
    src_pe = PositionalEmbeddings(seq_len, d_model, dropout_rate=0.3)
    # target embeddings
    tgt_input_embd = InputEmbeddings(tgt_vocab_size, d_model)
    tgt_pe = PositionalEmbeddings(seq_len, d_model, 0.3)
    # Encoder-Decoder
    encoder = TransformerEncoder(L=L_enc, d_model=d_model, h=h, d_ff=d_ff, dropout_rate=0.3, norm_first=norm_first)
    decoder = TransformerDecoder(L=L_dec, d_model=d_model, h=h, d_ff=d_ff, dropout_rate=0.3, norm_first=norm_first)
    # Trasnformers model
    transformer = Transformer(encoder, decoder, src_input_embd, src_pe, tgt_input_embd, tgt_pe)
    # initialize the parameters
    for parameter in transformer.parameters():
        if parameter.ndim > 1:
            nn.init.xavier_uniform_(parameter)
    return transformer


# implementing the transformers model
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, src_pe, tgt_embed, tgt_pe):
        super(Transformer, self).__init__()
        self.input_embeddings_src = src_embed
        self.positional_embeddings_src = src_pe
        self.input_embeddings_tgt = tgt_embed
        self.positional_embeddings_tgt = tgt_pe
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(tgt_embed.d_model, tgt_embed.vocab_size)

    def encode(self, x, src_mask):
        x = self.input_embeddings_src(x)
        x = self.positional_embeddings_src(x)
        return self.encoder(x, src_mask)

    def decode(self, x, memory, src_mask, tgt_mask):
        x = self.input_embeddings_tgt(x)
        x = self.positional_embeddings_tgt(x)
        return self.decoder(x, memory, src_mask, tgt_mask)
    
    def project(self, x):
        return self.linear(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder
        encoder_output = self.encode(src, src_mask)
        # decoder
        x = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        # projection
        x = self.project(x)
        return x
    
