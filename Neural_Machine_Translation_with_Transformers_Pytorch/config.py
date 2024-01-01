from dataclasses import dataclass

@dataclass
class GeneralConfig:
    lang_src: str = "en"
    lang_tgt: str = "it"
    train_size: float = 0.85
    val_size: float = 0.15
    seq_len: int = 325
    model_checkpoints_dir: str = "learned_weights"
    model_basename: str = f"transfromers_{lang_src}_{lang_tgt}"
    tokenizer_file: str = "Tokenizer_{0}.json"
    experiments_name: str = "runs/transfomers_model"

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 25
    preload: bool = None

@dataclass
class ModelConfig:
    d_model: int = 512
    num_attention_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    pre_LN: bool = True
