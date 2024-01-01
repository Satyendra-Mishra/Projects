import os
from datasets import load_dataset
import torch


def get_dataset(config, subset):
    raw_ds = load_dataset("opus_books", f'{config.lang_src}-{config.lang_tgt}', split=subset)
    return raw_ds

def save_model_states(config, epoch, model, optimizer, global_step):
    model_filename = f'{config.model_basename}_{epoch}.pt'
    save_dir = os.path.join(config.model_checkpoints_dir, model_filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step
    }, save_dir)