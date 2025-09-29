from preprocessor import load_and_preprocess
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from qwen import load_qwen

# This function takes in the path to the data file and extracts prey and predator values
with h5py.File("lotka_volterra_data.h5", "r") as f:
    prey = f["trajectories"][:, :, 0]
    predator = f["trajectories"][:, :, 1]

# Load the textual sequences from the dataset
all_seq = load_and_preprocess("lotka_volterra_data.h5")
model, tokenizer = load_qwen()

print("Finished load and preprocessing sequences!")

# Tokenize each sequence
token_list = [] 
for seq in all_seq:
    token = tokenizer(seq, return_tensors="pt", truncation=True)
    token_list.append(token)
    
print("Tokenized all sequences.")

# Select 2 random sequences to test generation
# This allows us to run predictions on a shuffled subset of the tokenized data
random.seed(423)
selected_indices = random.sample(range(len(token_list)), 2)
subset_tokens = [token_list[i] for i in selected_indices]
subset_real_prey = [prey[i] for i in selected_indices]
subset_real_predator = [predator[i] for i in selected_indices]

print("Selecting two random sequences to inspect preprocessed values and their tokens...")

print(f"Sequence number: {selected_indices[0]} has preprocessed values of: {all_seq[selected_indices[0]]}")
input_ids_1 = token_list[selected_indices[0]]['input_ids']
print(f"Sequence number: {selected_indices[0]} has token values of: {input_ids_1.tolist()[0]}")
print(f"Sequence number: {selected_indices[1]} has preprocessed values of: {all_seq[selected_indices[1]]}")
input_ids_2 = token_list[selected_indices[1]]['input_ids']
print(f"Sequence number: {selected_indices[1]} has token values of: {input_ids_2.tolist()[0]}")

print("generating the last twenty forecast values for these two sequences....")

# Run generation on selected prompts
pred_prey = []
pred_pred = []

for token in subset_tokens:
    # Select the first 80 prey-predator values (which are 960 tokens long)
    # THIS generates predictions for ALL the tokens. We select the portion we want.
    prompt = token["input_ids"][:, :960] 
    attention_mask = torch.ones_like(prompt)
    model.eval()
    output = model.generate(prompt, attention_mask=attention_mask, max_new_tokens=239)
    predicted_text = tokenizer.decode(output[0].tolist(), clean_up_tokenization_spaces=True)
            
    # SPLIT predicted pairs into prey and predator arrays
    predicted_pairs = predicted_text.split(";")
    predicted_prey = np.array([float(pair.split(",")[0].strip()) for pair in predicted_pairs if "," in pair])
    predicted_predator = np.array([float(pair.split(",")[1].strip()) for pair in predicted_pairs if "," in pair])
    
    pred_prey.append(predicted_prey)
    pred_pred.append(predicted_predator)

print("generation complete! Here are the 80 prompts + 20 forecasted values:")

print(f"sequence {selected_indices[0]} has the following forecasted prey values: {pred_prey[0]}") 

print(f"sequence {selected_indices[1]} has the following forecasted prey values: {pred_prey[1]}") 

print("Plotting first graph...")

# Plot the first system's true vs predicted populations
plt.figure(figsize=(8, 6))
time_steps = np.arange(0, 100)

# Plot predicted prey and predator values
plt.plot(time_steps, subset_real_prey[0], color='blue', label='True Prey', linestyle='-')
plt.plot(time_steps, subset_real_predator[0], color='red', label='True Predator', linestyle='-')
plt.plot(time_steps, pred_prey[0].squeeze() * 2.53, color='blue', label='Predicted Prey)', linestyle='--')
plt.plot(time_steps, pred_pred[0].squeeze() * 2.53, color='red', label='Predicted Predator)', linestyle='--')

plt.xlabel('Time Step')
plt.ylabel('Population')
plt.title(f"Comparing Qwen Predicted and True Prey and Predator Population Over Time for System {selected_indices[0]}",fontsize=14)
plt.legend()
plt.tight_layout()

print("Plotting second graph...")

# Plot the first system's true vs predicted populations
plt.figure(figsize=(8, 6))
time_steps = np.arange(0, 100)

# Plot predicted prey and predator values
plt.plot(time_steps, subset_real_prey[1], color='blue', label='True Prey', linestyle='-')
plt.plot(time_steps, subset_real_predator[1], color='red', label='True Predator', linestyle='-')
plt.plot(time_steps, pred_prey[1].squeeze() * 2.53, color='blue', label='Predicted Prey)', linestyle='--')
plt.plot(time_steps, pred_pred[1].squeeze() * 2.53, color='red', label='Predicted Predator)', linestyle='--')

plt.xlabel('Time Step')
plt.ylabel('Population')
plt.title(f"Comparing Qwen Predicted and True Prey and Predator Population Over Time for System {selected_indices[1]}",fontsize=14)
plt.legend()
plt.tight_layout()

print("Calculating MSE and MAPE loss...")

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


prey_330_mse = mseloss(np.array(subset_real_prey[0][80:]), np.array(pred_prey[0][80:] * 2.53))
pred_330_mse = mseloss(np.array(subset_real_predator[0][80:]), np.array(pred_pred[0][80:] * 2.53))
prey_330_mape = mapeloss(np.array(subset_real_prey[0][80:]), np.array(pred_prey[0][80:] * 2.53))
pred_330_mape = mapeloss(np.array(subset_real_predator[0][80:]), np.array(pred_pred[0][80:] * 2.53))

prey_167_mse = mseloss(np.array(subset_real_prey[1][80:]), np.array(pred_prey[1][80:] * 2.53))
pred_167_mse = mseloss(np.array(subset_real_predator[1][80:]), np.array(pred_pred[1][80:] * 2.53))
prey_167_mape = mapeloss(np.array(subset_real_prey[1][80:]), np.array(pred_prey[1][80:] * 2.53))
pred_167_mape = mapeloss(np.array(subset_real_predator[1][80:]), np.array(pred_pred[1][80:] * 2.53))

print(f"System 330: Prey MSE: {prey_330_mse}, Predator MSE: {pred_330_mse}, Prey MAPE: {prey_330_mape}, Predator MAPE: {pred_330_mape}")
print(f"System 167: Prey MSE: {prey_167_mse}, Predator MSE: {pred_167_mse}, Prey MAPE: {prey_167_mape}, Predator MAPE: {pred_167_mape}")

