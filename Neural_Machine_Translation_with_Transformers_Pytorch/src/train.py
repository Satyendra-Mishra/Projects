from pathlib import Path
import torch
from tqdm import tqdm
from .utils import save_model_states



def train_model(config, training_config, model, train_loader, val_loader, loss_fn, 
                optim, device, summary_writer, src_tokenizer, tgt_tokenizer):
    Path(config.model_checkpoints_dir).mkdir(parents=True, exist_ok=True)

    initial_epoch = 0
    global_step = 0
    # preload from a previously trained epoch if exists
    if training_config.preload:
        model_filename = f'{config.model_checkpoints_dir}_{training_config.preload}.pt'  # preload from this epoch
        print(f"Preloading model state from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optim.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # Start the training
    train_losses, val_losses = [], []
    for epoch in range(initial_epoch, training_config.epochs):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch:02d}")
        epoch_loss = 0
        for batch in batch_iterator:
            # input tensors
            encoder_input = batch["encoder_input"].to(device)    # (batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device)    # (batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)      # (batch_size, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)      # (batch_size, 1, seq_len, seq_len)
            # Feed the tensors to the model
            model_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask) # (batch_size, seq_len, tgt_vocab_size)
            # get the true labels
            labels = batch["label"].to(device) # (batch_size, seq_len)
            # compute the loss
            loss = loss_fn(model_output.view(-1, tgt_tokenizer.get_vocab_size()), labels.view(-1))
            epoch_loss += loss.item()
            batch_iterator.set_postfix({"loss": f'{loss.item(): 6.3f}'})
            # log the loss
            summary_writer.add_scalar("train_loss", loss.item(), global_step)
            summary_writer.flush()
            # backpropagate the loss
            loss.backward()
            # update the weights
            optim.step()
            optim.zero_grad()

            global_step += 1
        # run validation loop
        val_loss = run_validation(model, val_loader, loss_fn, device, global_step, summary_writer, src_tokenizer, tgt_tokenizer)
        val_losses.append(val_loss)
        train_losses.append(epoch_loss/len(train_loader))
        epoch_loss = 0
        # save the state after end of the epoch
        save_model_states(config, epoch, model, optim, global_step)
    return train_losses, val_losses


@torch.no_grad()
def run_validation(model, val_loader, loss_fn, device, summary_writer, src_tokenizer, tgt_tokenizer):
    model.eval()
    val_loss = 0
    for batch in tqdm(val_loader):
        # input tensors
        encoder_input = batch["encoder_input"].to(device)    # (batch_size, seq_len)
        decoder_input = batch["decoder_input"].to(device)    # (batch_size, seq_len)
        encoder_mask = batch["encoder_mask"].to(device)      # (batch_size, 1, 1, seq_len)
        decoder_mask = batch["decoder_mask"].to(device)      # (batch_size, 1, seq_len, seq_len)
        # Feed the tensors to the model
        model_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask) # (batch_size, seq_len, tgt_vocab_size)
        # get the true labels
        labels = batch["label"].to(device) # (batch_size, seq_len)
        # compute the loss
        loss = loss_fn(model_output.view(-1, tgt_tokenizer.get_vocab_size()), labels.view(-1))
        val_loss += loss.item()
        # log the loss
        summary_writer.add_scalar("val_loss", loss.item())
        summary_writer.flush()
    return val_loss/len(val_loader)