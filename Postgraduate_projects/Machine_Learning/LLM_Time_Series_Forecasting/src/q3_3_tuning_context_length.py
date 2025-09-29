# === Context Length Sweep Version ===

import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm
import wandb
from qwen import load_qwen
from lora_skeleton import LoRALinear
from flops import flops
from preprocessor import load_and_preprocess


def train_with_config(context_length, learning_rate=1e-4, lora_rank=8, run_group="M2_3B_train_ctx_300seq"):
    """

    Train a Qwen model with LoRA modifications using specified hyperparameters.

    This function loads a fresh Qwen model, applies Low-Rank Adaptation (LoRA) to the 
    attention projection layers, processes training/validation/test datasets into token IDs,
    prepares dataloaders, wraps training with HuggingFace's `Accelerator`, and logs training 
    metrics to Weights & Biases (WandB). It also evaluates the model on a held-out test set 
    using cross-entropy loss.

    Args:
        context_length (int): The maximum context length for the model.
        learning_rate (float): The learning rate used by the Adam optimizer.
        lora_rank (int): Rank parameter `r` for the LoRA layers.
        run_group (str): WandB run group/project name used for logging.

    Returns:
        float: Average cross-entropy loss on the test dataset, rounded to 3 decimal places.
    """
    
    print(f"Running config: Context Length={context_length}, LR={learning_rate}, Rank={lora_rank}")

    print("Loading fresh Qwen model...")
    model, tokenizer = load_qwen()
    model.to(device)

    print(f"Applying LoRA with rank={lora_rank}...")
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    print("Processing sequences...")
    val_input_ids = LoRALinear.process_sequences(val_texts, tokenizer, context_length, stride=context_length)
    test_input_ids = LoRALinear.process_sequences(test_texts, tokenizer, context_length, stride=context_length)
    train_input_ids = LoRALinear.process_sequences(train_texts, tokenizer, context_length, stride=context_length // 2)

    val_loader = DataLoader(TensorDataset(val_input_ids), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_input_ids), batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(TensorDataset(train_input_ids), batch_size=batch_size, shuffle=True)

    print(f"Initializing optimizer with learning rate {learning_rate}...")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    print("Preparing model with Accelerator...")
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    print("Logging into WandB and starting new run...")
    wandb.login()
    run = wandb.init(
        entity="jn492-university-of-cambridge",
        project=run_group,
        config={"learning_rate": learning_rate, "lora_rank": lora_rank, "context_length": context_length},
    )

    print("Begin training!")
    model.train()
    steps = 0
    while steps < 500:
        progress_bar = tqdm(train_loader, desc=f"Training Steps {steps}")
        for (batch,) in progress_bar:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(model.device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1

            if steps % 10 == 0:
                model.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for (val_batch,) in val_loader:
                        val_batch = val_batch.to(model.device)
                        val_outputs = model(val_batch, labels=val_batch)
                        val_loss_total += val_outputs.loss.item()
                val_loss = val_loss_total / len(val_loader)
                progress_bar.set_postfix(train_loss=loss.item(), val_loss=val_loss)
                run.log({"train loss": loss.item(), "val loss": val_loss})
            else:
                progress_bar.set_postfix(train_loss=loss.item())
                run.log({"train loss": loss.item()})

            if steps >= 500:
                break

    print("Finished training! Starting cross entropy evaluation on test set...")
    model.eval()
    trained_loss = 0.0
    with torch.no_grad():
        for (test_batch,) in test_loader:
            test_batch = test_batch.to(device)
            outputs = model(test_batch, labels=test_batch)
            trained_loss += outputs.loss.item()

    avg_loss_trained = trained_loss / len(test_loader)
    print(f"Test Cross Entropy Loss for Context Length={context_length}: {avg_loss_trained:.3f}")
    run.log({"test cross entropy loss": avg_loss_trained})

    run.finish()
    print(f"Finished run: Context Length={context_length}\n")
    
    return avg_loss_trained


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Importing libraries and loading dataset...")

    with h5py.File("lotka_volterra_data.h5", "r") as f:
        prey = f["trajectories"][:, :, 0]
        predator = f["trajectories"][:, :, 1]

    print("Preprocessing Lotka-Volterra sequences...")
    all_seq = load_and_preprocess("lotka_volterra_data.h5")

    print("Sampling and splitting data...")
    random.seed(423)
    sampled_indices = random.sample(range(len(all_seq)), 300)
    subset_seq = [all_seq[i] for i in sampled_indices]

    train_texts, remaining_texts, train_indices, remaining_indices = train_test_split(
        subset_seq, sampled_indices, train_size=0.7, test_size=0.3, random_state=423
    )
    val_texts, test_texts, val_indices, test_indices = train_test_split(
        remaining_texts, remaining_indices, train_size=0.5, test_size=0.5, random_state=423
    )

    batch_size = 4
    context_lengths = [128, 512, 768]
    test_losses = []

    for ctx_len in context_lengths:
        loss = train_with_config(context_length=ctx_len)
        test_losses.append(loss)

    best_ctx_index = int(np.argmin(test_losses))
    best_ctx_len = context_lengths[best_ctx_index]
    print(f"Best context length: {best_ctx_len} with test loss {test_losses[best_ctx_index]:.3f}")
