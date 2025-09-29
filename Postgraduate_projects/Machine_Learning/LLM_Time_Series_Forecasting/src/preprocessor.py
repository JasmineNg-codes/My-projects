import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from qwen import load_qwen

def load_and_preprocess(file_path):
    """
    Loads and preprocesses the prey-predator dataset for Qwen.

    Steps:
    1. Loads the prey and predator time series data from an HDF5 file.
    2. Scales the prey and predator values so that most are in the range [0, 1]:
       - Prey and predator divided by a scaling factor (90th percentile of all values).
    3. Converts the scaled values into strings with 3 decimal precision.
    4. Combines prey and predator values into a single string per sequence, 
       where each time step is formatted as "prey,pred" and time steps are joined with semicolons:
       Example: "0.432,0.214;0.531,0.198;...;0.876,0.320"
    5. Stores all sequences as a list of 1000 strings (one per sequence).

    Args:
        file_path (str): Path to the HDF5 file containing the dataset.

    Returns:
        List[str]: A list of 1000 string sequences (prey-predator pairs over time).
    """
    with h5py.File(file_path, "r") as f:
        prey = f["trajectories"][:, :, 0]      # Shape: (1000, 100)
        predator = f["trajectories"][:, :, 1]  # Shape: (1000, 100)

    
    # Flatten the arrays to compute percentiles over the full dataset
    all_values = np.concatenate([prey.flatten(), predator.flatten()])

    # Compute the 90th percentile
    alpha_percentile_value = np.percentile(all_values, 90)  # This becomes your scaling factor

    # Apply rescaling
    prey_scaled = prey / alpha_percentile_value
    predator_scaled = predator / alpha_percentile_value

    # Convert to string format
    prey_strings = [[f"{val:.3f}" for val in system] for system in prey_scaled]
    predator_strings = [[f"{val:.3f}" for val in system] for system in predator_scaled]

    # Combine prey and predator into single sequences
    prey_pred_sequences = []
    for system in range(len(prey_strings)):
        system_sequence = [
            f"{prey_strings[system][t]},{predator_strings[system][t]}" for t in range(len(prey_strings[system]))
        ]
        prey_pred_sequences.append(";".join(system_sequence))

    return prey_pred_sequences

