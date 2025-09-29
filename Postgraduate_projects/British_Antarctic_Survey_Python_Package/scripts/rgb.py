print("begin loading packages...")

import pandas as pd
import os
from DELPHI.LPdetection import LPtrain
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))        
project_root = os.path.abspath(os.path.join(script_dir, "..")) 

# Hard (T1) and Soft (T2) paths
impath_hard = './data/hard/images'
lppath_hard = './data/hard/annotations'

impath_soft = './data/soft/images'
lppath_soft = './data/soft/annotations'

# Train both sets
print("train hard...")
trainer_hard = LPtrain(impath=impath_hard, lppath=lppath_hard, num_images=None)

print("train soft...")
trainer_soft = LPtrain(impath=impath_soft, lppath=lppath_soft, num_images=None)

image_dict_hard = trainer_hard.image_dict
image_dict_soft = trainer_soft.image_dict

def plot_laserpoint_rgb_stats(image_dict_hard, image_dict_soft, results_dir):
    transect_colors = {"T1": [], "T2": []}  # T1 = hard, T2 = soft

    print("Processing Hard substrates")
    # Process hard (T1)
    for image_id, data in image_dict_hard.items():
        img = data["image"]
        lps = data["laser_points"]
        for x, y in lps:
            if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                rgb = img[int(y), int(x), :]
                transect_colors["T1"].append(rgb)

    # Process soft (T2)
    
    print("Processing Soft substrates")
    for image_id, data in image_dict_soft.items():
        img = data["image"]
        lps = data["laser_points"]
        for x, y in lps:
            if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                rgb = img[int(y), int(x), :]
                transect_colors["T2"].append(rgb)

    t1_colors = np.array(transect_colors["T1"])
    t2_colors = np.array(transect_colors["T2"])

    print("Calculating means")
    means = {
        "T1": np.mean(t1_colors, axis=0),
        "T2": np.mean(t2_colors, axis=0)
    }
    
    print("Calculating stdev")
    stds = {
        "T1": np.std(t1_colors, axis=0),
        "T2": np.std(t2_colors, axis=0)
    }

    channels = ["Red", "Green", "Blue"]
    x = np.arange(len(channels))
    width = 0.35

    print("Plotting figure")
    fig, ax = plt.subplots()
    ax.bar(x - width/2, means["T1"], width, yerr=stds["T1"], label="Hard (T1)", capsize=5)
    ax.bar(x + width/2, means["T2"], width, yerr=stds["T2"], label="Soft (T2)", capsize=5)

    ax.set_ylabel("Mean RGB Value")
    ax.set_title("Laser Point RGB Comparison Between Transects")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend()
    fig.tight_layout()
    
    out_path = os.path.join(results_dir, "2.4_rgb.png")
    plt.savefig(out_path, dpi=300)
    print(f"RGB plot saved to: {out_path}")
    
# Setup results directory
results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)

# Call the plot function with both image_dicts
plot_laserpoint_rgb_stats(image_dict_hard, image_dict_soft, results_dir)
