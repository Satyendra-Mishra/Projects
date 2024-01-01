from pathlib import Path
import tokenizers


# An iterator to get all the senteces in the language corpus
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

# Build a tokenizer on the language corpus
def build_tokenizer(config, ds, lang):
    # special tokens
    spl_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    # path to tokenizer file
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    # if no existing tokenizer file build a new one
    if not Path.exists(tokenizer_path):
        print("Building Tokenizer from the language =", lang)
        # tokenizer model from tokenizers lib
        tokenizer_model = tokenizers.models.WordLevel(unk_token="[UNK]")
        # create a tokenizer onbject from the chosen model
        tokenizer = tokenizers.Tokenizer(tokenizer_model)
        # pre_tokenizer before training
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        # trainer to tokenize the sentences
        trainer = tokenizers.trainers.WordLevelTrainer(special_tokens=spl_tokens, min_frequency=2)
        # train the tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print("Found existing tokenizer file at =", str(tokenizer_path), "\n Loading the tokenizer from file")
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_tokenizers(config, ds):
    tokenizer_src = build_tokenizer(config=config, ds=ds, lang=config.lang_src)
    tokenizer_tgt = build_tokenizer(config=config, ds=ds, lang=config.lang_tgt)
    return tokenizer_src, tokenizer_tgt