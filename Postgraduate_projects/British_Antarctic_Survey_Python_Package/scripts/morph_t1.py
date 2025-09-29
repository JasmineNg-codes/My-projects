import pandas as pd
import glob
import re
import os
from sklearn.model_selection import KFold
from DELPHI.LPdetection import LPPreprocess, LPtrain, LPDetect
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))        # e.g. /home/jn492/jn492/scripts
project_root = os.path.abspath(os.path.join(script_dir, "..")) # e.g. /home/jn492/jn492

og_im = "./data_4/t1/images" #Just soft substrate, change this path to desktop path to images 
og_lp = "./data_4/t1/annotations"

LPPreprocess(impath = og_im, lppath = og_lp, validation = True, train_ratio = 0.8, seed=100)

train_impath='./data_4/t1/train_images'
train_lppath='./data_4/t1/train_annotations'

val_impath='./data_4/t1/val_images'
val_lppath='./data_4/t1/val_annotations'

results = []

#### TRAINING 
trainer = LPtrain(impath = train_impath, lppath = train_lppath, num_images = None)
combined_mask = trainer.spatial_learning(lpradius=25, plot = False)

threshold, cluster_gamma, S_gamma, _ = trainer.color_learning(lpradius=3, backgroundradius=25, k=7)

#storing these values in the results for future analysis
S_gamma = np.array(S_gamma)
red_values = S_gamma[:, 0]
red_min = red_values.min()
red_max = red_values.max()
red_range = red_max - red_min

#### TESTING
detector = LPDetect(impath = val_impath, lppath = val_lppath, num_images = None, threshold = threshold, cluster_gamma = cluster_gamma, S_gamma = S_gamma)
detector.gray_value_image(plot=False)

kernels = [1,2,3,4,5,10,20]

for kernel in kernels:
    detector.binary_mask_image(morphology = True, kernel_dim = kernel, plot=False)
    detector.apply_train_mask(combined_mask, plot=False)
    detector.connect_and_weight_regions(plot=False)
    detector.predict_laserpoint(trainer.image_dict)
    #detector.plot_predictions()
    precision, recall, f1, _ = detector.performance_test()

    #storing all results
    results.append({
        "kernel size": kernel,
        "threshold": threshold,
        "num_curated_laser_point_colors": len(S_gamma),
        "red_min": red_min,
        "red_max": red_max,
        "red_range": red_range,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })
    
df_results = pd.DataFrame(results)
results_dir = os.path.join(script_dir, "..", "results","morph_t1")  # correct relative path
results_dir = os.path.abspath(results_dir)  # convert to absolute path
os.makedirs(results_dir, exist_ok=True)     # ensure it exists
df_results.to_csv(os.path.join(results_dir, "morph_t1.csv"), index=False)
print("Saved to:", os.path.join(results_dir, "morph_t1.csv"))

