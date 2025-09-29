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
import re


def train_with_config(context_length=512, learning_rate=1e-4, lora_rank=8, run_group="M2_3C_final_model"):
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
    while steps < 3000: #change to 
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

            if steps % 50 == 0: #change to 50
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

            if steps >= 3000:
                break

    print("Finished training! Starting cross entropy evaluation on test set...")
    model.eval()
    trained_loss = 0.0
    with torch.no_grad():
        for (test_batch,) in test_loader:
            test_batch = test_batch.to(model.device)
            outputs = model(test_batch, labels=test_batch)
            trained_loss += outputs.loss.item()

    avg_loss_trained = trained_loss / len(test_loader)
    print(f"Test Cross Entropy Loss for Context Length={context_length}: {avg_loss_trained:.3f}")
    run.log({"test cross entropy loss": avg_loss_trained})

    run.finish()
    torch.save(model.state_dict(), "/content/my_model.pt")
    print("Begin MSE/MAPE evaluation by doing model.generate() for test sets...")

    print("Start generation...")

    pred_prey = []
    pred_pred = []
    
    model.eval()

        # Generate predictions
    for k, sequence in enumerate(tqdm(test_texts, desc="Generating on full sequences")):
        system_index = test_indices[k]

        token = tokenizer(sequence, return_tensors="pt", truncation=True, add_special_tokens=False)
        prompt = token["input_ids"][:, :960].to(device)
        attention_mask = torch.ones_like(prompt).to(device)

        output = model.generate(prompt, attention_mask=attention_mask, max_new_tokens=239)
        decoded = tokenizer.decode(output[0], clean_up_tokenization_spaces=True)
        decoded = decoded.replace("[", "").replace("]", "").replace("<|endoftext|>", "").strip()

        if system_index == 133:
            print(f"\n===== DEBUG: SYSTEM {system_index} =====")
            print("Generated token IDs:")
            print(output[0].tolist())
            print("Decoded text:")
            print(decoded)

        pairs = decoded.split(";")
        valid_pairs = []
        for pair in pairs:
            if re.match(r"^\s*[-+]?\d*\.\d+,\s*[-+]?\d*\.\d+\s*$", pair):
                x_str, y_str = pair.split(",")
                valid_pairs.append((float(x_str.strip()), float(y_str.strip())))

        if not valid_pairs:
            pred_prey.append(np.array([]))
            pred_pred.append(np.array([]))
            continue

        predicted_prey = np.array([x for x, _ in valid_pairs])
        predicted_predator = np.array([y for _, y in valid_pairs])

        pred_prey.append(predicted_prey)
        pred_pred.append(predicted_predator)

    def mseloss(true,predicted):
        mse = np.mean((true - predicted)**2)
        return np.round(mse, 3)

    def mapeloss(true, predicted):
        true = np.array(true)
        predicted = np.array(predicted)
        
        # Avoid division by zero by adding a small epsilon where true == 0
        epsilon = 1e-8
        denominator = np.where(true == 0, epsilon, true)
        
        mape = np.mean(np.abs((true - predicted) / denominator)) * 100
        return np.round(mape, 3)

    print("start calculating MSE and MAPE...")

    print("Finished generation!")

    # Now continue with your existing metric code as-is,
    # but modify the evaluation loop to skip incomplete sequences:

    all_mse_prey = []
    all_mape_prey = []
    all_mse_pred = []
    all_mape_pred = []
    system_ids = []

    for k in range(len(pred_prey)):
        system_index = test_indices[k]

        true_prey = prey[system_index, -20:]
        true_pred = predator[system_index, -20:]

        predicted_prey = pred_prey[k][-20:] * 2.53
        predicted_pred = pred_pred[k][-20:] * 2.53

        if len(predicted_prey) < 20 or len(predicted_pred) < 20:
            print(f"Skipping System ID: {system_index} due to insufficient valid predictions")
            continue

        system_ids.append(system_index)
        mse_pre = mseloss(true_prey, predicted_prey)
        mape_pre = mapeloss(true_prey, predicted_prey)
        mse_pr = mseloss(true_pred, predicted_pred)
        mape_pr = mapeloss(true_pred, predicted_pred)

        all_mse_prey.append(mse_pre)
        all_mape_prey.append(mape_pre)
        all_mse_pred.append(mse_pr)
        all_mape_pred.append(mape_pr)
        
    for i in range(len(system_ids)):
        print(f"System ID: {system_ids[i]} has MSE prey: {all_mse_prey[i]}, "
            f"MAPE prey: {all_mape_prey[i]}, MSE predator: {all_mse_pred[i]}, "
            f"MAPE predator: {all_mape_pred[i]}")
    
    avg_mse_prey = np.mean(all_mse_prey)
    avg_mape_prey = np.mean(all_mape_prey)
    avg_mse_pred = np.mean(all_mse_pred)
    avg_mape_pred = np.mean(all_mape_pred)
    
    print("Average MSE (Prey): {:.3f}".format(avg_mse_prey))
    print("Average MAPE (Prey): {:.3f}".format(avg_mape_prey))
    print("Average MSE (Predator): {:.3f}".format(avg_mse_pred))
    print("Average MAPE (Predator): {:.3f}".format(avg_mape_pred))

    print("Finally, plot the second graph to check!")

    system_index = test_indices[1]

    true_prey = prey[system_index]
    true_pred = predator[system_index]

    # Step 2: Get time steps
    time_steps = np.arange(0, 100)

    # Step 3: Create plot
    plt.figure(figsize=(8, 6))

    # True full values
    plt.plot(time_steps, true_prey, color='blue', label='True Prey', linestyle='-')
    plt.plot(time_steps, true_pred, color='red', label='True Predator', linestyle='-')

    # Predicted values (scaled)
    plt.plot(time_steps, pred_prey[1]*2.53, color='blue', label='Predicted Prey', linestyle='--')
    plt.plot(time_steps, pred_pred[1]*2.53, color='red', label='Predicted Predator', linestyle='--')

    # Labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Population')
    plt.title(f"Qwen Predictions vs. Ground Truth for System {system_index}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("q3_final_model_generate.png")

    return {
    "cross_entropy_loss": avg_loss_trained,
    "avg_mse_prey": avg_mse_prey,
    "avg_mape_prey": avg_mape_prey,
    "avg_mse_pred": avg_mse_pred,
    "avg_mape_pred": avg_mape_pred
}


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
    sampled_indices = random.sample(range(len(all_seq)), 1000)
    subset_seq = [all_seq[i] for i in sampled_indices]

    train_texts, remaining_texts, train_indices, remaining_indices = train_test_split(
        subset_seq, sampled_indices, train_size=0.7, test_size=0.3, random_state=423
    )
    val_texts, test_texts, val_indices, test_indices = train_test_split(
        remaining_texts, remaining_indices, train_size=0.5, test_size=0.5, random_state=423
    )

    batch_size = 4
    
    loss = train_with_config(context_length=512, learning_rate=1e-4, lora_rank=8, run_group="M2_3C_final_model")
    print(loss)
    
