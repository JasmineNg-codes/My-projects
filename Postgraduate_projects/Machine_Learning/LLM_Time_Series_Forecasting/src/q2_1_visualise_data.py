import numpy as np
import h5py
import matplotlib.pyplot as plt
from qwen import load_qwen
import torch

print("Reading in data...")

with h5py.File("lotka_volterra_data.h5", "r") as f:

    print("time shape:", f["time"].shape)
    print("trajectory shape:", f["trajectories"].shape)
    print(f"First five entries of the variables prey, {f['trajectories'][1,0:5,0]}, and predator {f['trajectories'][1,0:5,1]}")

    prey = f["trajectories"][:,:,0]
    predator = f["trajectories"][:,:,1]

prey_flatten=prey.flatten()
predator_flatten=predator.flatten()

print("Plotting data...")

fig, ax  = plt.subplots(1,2,figsize=(10,5))
ax[0].hist(prey_flatten,bins=100, color='blue')
ax[0].set_title('Prey')
ax[0].set_xlabel('Population density')
ax[0].set_ylabel('Frequency')
ax[1].hist(predator_flatten,bins=100,color='red')
ax[1].set_title('Predator')
ax[1].set_xlabel('Population density')
ax[1].set_ylabel('Frequency')


plt.show(); #semicolon to suppress output
plt.tight_layout()