print("importing libraries...")

import numpy as np
import h5py
import matplotlib.pyplot as plt
from qwen import load_qwen
import torch
from lora_skeleton import LoRALinear
from accelerate import Accelerator
from flops import flops
from preprocessor import load_and_preprocess
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading in data...")

# This function takes in the path to the data file and extracts prey and predator values
with h5py.File("lotka_volterra_data.h5", "r") as f:
    prey = f["trajectories"][:, :, 0]
    predator = f["trajectories"][:, :, 1]

print("preprocessing...")

# Load the textual sequences from the dataset
all_seq = load_and_preprocess("lotka_volterra_data.h5")
model, tokenizer = load_qwen()
model.to(device)

lora_rank = 4

print("Wrapping LoRA around qwen model, such that rank=4.")

# Actually apply LoRA to the model:
for layer in model.model.layers:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)
    
random.seed(423)
sampled_indices = random.sample(range(len(all_seq)), 100)
subset_seq = [all_seq[i] for i in sampled_indices]

print("Train, val, test split...")
# First split: 70% train, 30% (val + test)
train_texts, remaining_texts, train_indices, remaining_indices = train_test_split(
    subset_seq,
    sampled_indices,
    train_size=0.7,
    test_size=0.3,
    random_state=423
)

# Second split: 1/3 val, 2/3 test from the 30
val_texts, test_texts, val_indices, test_indices = train_test_split(
    remaining_texts,
    remaining_indices,
    train_size=1/2,
    test_size=1/2,
    random_state=423
)

# Defines the maximum context length
max_ctx_length = 512
batch_size = 4

val_input_ids = LoRALinear.process_sequences(
    val_texts, tokenizer, max_ctx_length, stride=max_ctx_length
)

val_dataset = TensorDataset(val_input_ids)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_input_ids = LoRALinear.process_sequences(
    test_texts, tokenizer, max_ctx_length, stride=max_ctx_length
)

test_dataset = TensorDataset(test_input_ids)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Begin cross entropy evaluation using test set...")

untrained_model, untrained_tokenizer = model, tokenizer
untrained_model.to(device)

untrained_model.eval()

untrained_loss = 0.0

#give whole sequence, guess one by one
with torch.no_grad(): #disable gradient update
    for (test_batch,) in tqdm(test_loader, desc="Evaluating", total=len(test_loader)):
        test_batch = test_batch.to(device)
        outputs = untrained_model(test_batch, labels=test_batch)
        untrained_loss += outputs.loss.item()

avg_loss_untrained = untrained_loss / len(test_loader)

print(f"Untrained model test loss: {avg_loss_untrained:.3f}")

print("Finished cross entropy evaluation!")

print("Begin MSE/MAPE evaluation by doing model.generate() for test sets...")

print("Start generation...")

#generate 20 values on test set
pred_prey = []
pred_pred = []
untrained_model.eval()

for sequence in tqdm(test_texts, desc="Generating on full sequences"):
    # Get input tensor and ensure it's batched
    token = tokenizer(sequence, return_tensors="pt", truncation=True, add_special_tokens=False)
    prompt = token["input_ids"][:, :960].to(device)
    attention_mask = torch.ones_like(prompt).to(device)
    output = untrained_model.generate(prompt, attention_mask=attention_mask, max_new_tokens=239)
    predicted_text = tokenizer.decode(output[0].tolist(), clean_up_tokenization_spaces=True)
    predicted_text = predicted_text.replace("]", "").replace("[", "").strip()
    predicted_pairs = predicted_text.split(";")
    predicted_prey = np.array([float(pair.split(",")[0].strip()) for pair in predicted_pairs if "," in pair])
    predicted_predator = np.array([float(pair.split(",")[1].strip()) for pair in predicted_pairs if "," in pair])
    
    pred_prey.append(predicted_prey)
    pred_pred.append(predicted_predator)

print("Finished generation!")

def mseloss(true,predicted):
    """
    Args:
        true: true value
        predicted : predicted value / model output value

    Returns:
        float: mean squared error rounded to 3 decimal places.
    """
    mse = np.mean((true - predicted)**2)
    return np.round(mse, 3)

def mapeloss(true, predicted):
    """
    Args:
        true (_type_): true value
        predicted (_type_): predicted value / model output value

    Returns:
        float: mean absolute percentage error rounded to 3 decimal places.
    """
    true = np.array(true)
    predicted = np.array(predicted)
    
    # Avoid division by zero by adding a small epsilon where true == 0
    epsilon = 1e-8
    denominator = np.where(true == 0, epsilon, true)
    
    mape = np.mean(np.abs((true - predicted) / denominator)) * 100
    return np.round(mape, 3)

print("start calculating MSE and MAPE...")


# Store results
all_mse_prey = []
all_mape_prey = []
all_mse_pred = []
all_mape_pred = []
system_ids = []

# Loop through each validation system
for k in range(len(pred_prey)):
    system_index = test_indices[k]
    system_ids.append(system_index)

    # True values from original data
    true_prey = prey[system_index, -20:]
    true_pred = predator[system_index, -20:]

    # Model predictions
    predicted_prey = pred_prey[k][-20:] * 2.53
    predicted_pred = pred_pred[k][-20:] * 2.53

    # Compute metrics
    mse_pre = mseloss(true_prey, predicted_prey)
    mape_pre = mapeloss(true_prey, predicted_prey)

    mse_pr = mseloss(true_pred, predicted_pred)
    mape_pr = mapeloss(true_pred, predicted_pred)

    # Store results
    all_mse_prey.append(mse_pre)
    all_mape_prey.append(mape_pre)
    all_mse_pred.append(mse_pr)
    all_mape_pred.append(mape_pr)
    
for i in range(len(system_ids)):
    print(f"System ID: {system_ids[i]} has MSE prey: {all_mse_prey[i]}, "
          f"MAPE prey: {all_mape_prey[i]}, MSE predator: {all_mse_pred[i]}, "
          f"MAPE predator: {all_mape_pred[i]}")
    
print("Average MSE (Prey): {:.3f}".format(np.mean(all_mse_prey)))
print("Average MAPE (Prey): {:.3f}".format(np.mean(all_mape_prey)))
print("Average MSE (Predator): {:.3f}".format(np.mean(all_mse_pred)))
print("Average MAPE (Predator): {:.3f}".format(np.mean(all_mape_pred)))

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
plt.savefig("q3_second_graph_untrained_model.png")



