import torch

class NMTDataset(torch.utils.data.Dataset):
    def __init__(self, ds, seq_len, lang_src, lang_tgt, tokenizer_src, tokenizer_tgt):
        self.ds = ds
        self.seq_len = seq_len
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # special tokens
        self.sos_token = torch.LongTensor([tokenizer_src.token_to_id("[SOS]")])
        self.eos_token = torch.LongTensor([tokenizer_src.token_to_id("[EOS]")])
        self.pad_token = torch.LongTensor([tokenizer_src.token_to_id("[PAD]")])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair["translation"][self.lang_src]
        tgt_text = src_tgt_pair["translation"][self.lang_tgt]

        encoder_input_token_ids = self.tokenizer_src.encode(src_text).ids
        decoder_input_token_ids = self.tokenizer_tgt.encode(tgt_text).ids

        num_padding_tokens_enc = self.seq_len - len(encoder_input_token_ids) - 2
        num_padding_tokens_dec = self.seq_len - len(decoder_input_token_ids) - 1

        if num_padding_tokens_enc < 0:
            raise ValueError("Input sentence for encoder is too long.")
        if num_padding_tokens_dec < 0:
            raise ValueError("Input sentence for decoder is too long.")

        # Encoder input
        encoder_input = torch.cat([self.sos_token,
                                   torch.LongTensor(encoder_input_token_ids),
                                   self.eos_token,
                                   torch.LongTensor([self.pad_token]*num_padding_tokens_enc)])
        # Decoder input
        decoder_input = torch.cat([self.sos_token,
                                   torch.LongTensor(decoder_input_token_ids),
                                   torch.LongTensor([self.pad_token]*num_padding_tokens_dec)])
        # Label (decoder output)
        label = torch.cat([torch.LongTensor(decoder_input_token_ids),
                           self.eos_token,
                           torch.LongTensor([self.pad_token]*num_padding_tokens_dec)])

        assert encoder_input.shape[0] == self.seq_len, "Encoder input is shorter than seq_len"
        assert decoder_input.shape[0] == self.seq_len, "Decoder input is shorter than seq_len"
        assert label.shape[0] == self.seq_len,         "Label is shorter than the seq_len"

        packed_input = {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]), # (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

        return packed_input
    

def causal_mask(size):
    mask = torch.triu(torch.ones(size=(1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def build_dataset(config, raw_ds, src_tokenizer, tgt_tokenizer):
    # train-validation split
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(raw_ds, [config.train_size, config.val_size])
    # create torch datasets
    train_dataset = NMTDataset(train_ds_raw, config.seq_len, config.lang_src, config.lang_tgt, src_tokenizer, tgt_tokenizer)
    val_dataset = NMTDataset(val_ds_raw, config.seq_len, config.lang_src, config.lang_tgt, src_tokenizer, tgt_tokenizer)

    return train_dataset, val_dataset

def build_dataloader(dataset, training_config):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)
    return dataloader
