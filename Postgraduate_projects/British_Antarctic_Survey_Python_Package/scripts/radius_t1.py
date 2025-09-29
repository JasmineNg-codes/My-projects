import pandas as pd
import glob
import re
import os
from sklearn.model_selection import KFold
from DELPHI.LPdetection import LPPreprocess, LPtrain, LPDetect
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))        # e.g. /home/jn492/jn492/scripts
project_root = os.path.abspath(os.path.join(script_dir, "..")) # e.g. /home/jn492/jn492


# Output folders for plots
results_dir = os.path.join(project_root, "results")
results_dir = os.path.join(project_root, "results", "radius_t1") #store it under baseline_t1 results
kmeans_dir = os.path.join(results_dir, "kmeans_plots")
os.makedirs(kmeans_dir, exist_ok=True)

og_im = "./data_5/t1/images" #Just soft substrate, change this path to desktop path to images 
og_lp = "./data_5/t1/annotations"

LPPreprocess(impath = og_im, lppath = og_lp, validation = True, train_ratio = 0.8, seed=100)

train_impath='./data_5/t1/train_images'
train_lppath='./data_5/t1/train_annotations'

val_impath='./data_5/t1/val_images'
val_lppath='./data_5/t1/val_annotations'

print(f"Number of training images: {len(glob.glob(os.path.join(train_impath, '*')))}")
print(f"Number of validation images: {len(glob.glob(os.path.join(val_impath, '*')))}")

lpcolor = [1,3,5,10,15,30,50] #radius
backgroundcolor = [10, 20, 25, 40, 60, 80, 100] #radius

trainer = LPtrain(impath = train_impath, lppath = train_lppath, num_images = None)

results = []

for i in lpcolor:
    print(f"Trying radius {i} for laser point color clustering")
    for j in backgroundcolor: 
        if j <= i:
            continue  # skip if background radius is not larger than LP radius
        print(f"Trying radius {j} for background color clustering")
        combined_mask = trainer.spatial_learning(lpradius=j, plot = False)
        
            # Save interactive Plotly figure as .html instead of .png
        plot_path = os.path.join(kmeans_dir, f"t1_kmean_lp{i}_bkgd{j}.html")
        
        threshold, cluster_gamma, S_gamma, fig = trainer.color_learning(lpradius=i, backgroundradius=j, k=7, plot=True)

            # Save interactive Plotly figure as .html instead of .png
        if fig is not None:
            fig.write_html(plot_path)
            print(f"Saved interactive k-means plot to {plot_path}")

        #storing these values in the results for future analysis
        S_gamma = np.array(S_gamma)
        red_values = S_gamma[:, 0]
        red_min = red_values.min()
        red_max = red_values.max()
        red_range = red_max - red_min

        #### TESTING
        detector = LPDetect(impath = val_impath, lppath = val_lppath, num_images = None, threshold = threshold, cluster_gamma = cluster_gamma, S_gamma = S_gamma)
        detector.gray_value_image(plot=False)
        detector.binary_mask_image(plot=False)
        detector.apply_train_mask(combined_mask, plot=False)
        detector.connect_and_weight_regions(plot=False)
        detector.predict_laserpoint(trainer.image_dict)
        #detector.plot_predictions()
        precision, recall, f1, _ = detector.performance_test()

        #storing all results
        results.append({
            "LP_color_radius": i,
            "Background_color_radius": j,
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
results_dir = os.path.join(script_dir, "..", "results","radius_t1")  # correct relative path
results_dir = os.path.abspath(results_dir)  # convert to absolute path
os.makedirs(results_dir, exist_ok=True)     # ensure it exists
df_results.to_csv(os.path.join(results_dir, "radius_t1.csv"), index=False)
print("Saved to:", os.path.join(results_dir, "radius_t1.csv"))