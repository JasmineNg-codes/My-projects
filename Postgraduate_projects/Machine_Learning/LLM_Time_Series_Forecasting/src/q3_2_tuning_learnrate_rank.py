"""
This version of your script includes hyperparameter tuning over:
- Learning rate: [1e-5, 5e-5, 1e-4]
- LoRA rank: [2, 4, 8]

It ensures a fresh model is loaded for each run.
Results are logged to separate W&B projects: "3bi_runs" for LR tuning, "3bii_runs" for LoRA rank tuning.
"""

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


def train_with_config(learning_rate, lora_rank, run_group):
    """

    Train a Qwen model with LoRA modifications using specified hyperparameters.

    This function loads a fresh Qwen model, applies Low-Rank Adaptation (LoRA) to the 
    attention projection layers, processes training/validation/test datasets into token IDs,
    prepares dataloaders, wraps training with HuggingFace's `Accelerator`, and logs training 
    metrics to Weights & Biases (WandB). It also evaluates the model on a held-out test set 
    using cross-entropy loss.

    Args:
        learning_rate (float): The learning rate used by the Adam optimizer.
        lora_rank (int): Rank parameter `r` for the LoRA layers.
        run_group (str): WandB run group/project name used for logging.

    Returns:
        float: Average cross-entropy loss on the test dataset, rounded to 3 decimal places.

    """
    print(f"Running config: LR={learning_rate}, Rank={lora_rank}, Group={run_group}")

    print("Loading fresh Qwen model...")
    model, tokenizer = load_qwen()
    model.to(device)

    print(f"Applying LoRA with rank={lora_rank}...")
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    print("Processing sequences...")
    val_input_ids = LoRALinear.process_sequences(val_texts, tokenizer, max_ctx_length, stride=max_ctx_length)
    test_input_ids = LoRALinear.process_sequences(test_texts, tokenizer, max_ctx_length, stride=max_ctx_length)
    train_input_ids = LoRALinear.process_sequences(train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2)

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
        config={"learning_rate": learning_rate, "lora_rank": lora_rank},
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
    print(f"Test Cross Entropy Loss for LR={learning_rate}, Rank={lora_rank}: {avg_loss_trained:.3f}")
    run.log({"test cross entropy loss": avg_loss_trained})

    run.finish()
    print(f"Finished run: LR={learning_rate}, Rank={lora_rank}\n")
    
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
    sampled_indices = random.sample(range(len(all_seq)), 100)
    subset_seq = [all_seq[i] for i in sampled_indices]

    train_texts, remaining_texts, train_indices, remaining_indices = train_test_split(
        subset_seq, sampled_indices, train_size=0.7, test_size=0.3, random_state=423
    )
    val_texts, test_texts, val_indices, test_indices = train_test_split(
        remaining_texts, remaining_indices, train_size=0.5, test_size=0.5, random_state=423
    )

    max_ctx_length = 512
    batch_size = 4

    print("Starting learning rate sweep with fixed LoRA rank = 4...")
    
    learning_rates = [1e-5, 5e-5, 1e-4]
    test_losses = []
    
    for lr in learning_rates:
        loss = train_with_config(learning_rate=lr, lora_rank=4, run_group="M2_3B_train_LR")
        test_losses.append(loss)

    best_lr_index = int(np.argmin(test_losses))
    best_lr = learning_rates[best_lr_index]
    print(f"Best LR from sweep: {best_lr:.1e} with test loss {test_losses[best_lr_index]:.3f}")

    print("Starting LoRA rank sweep using best learning rate...")
    for rank in [2, 4, 8]:
        train_with_config(learning_rate=best_lr, lora_rank=rank, run_group="M2_3B_train_rank")