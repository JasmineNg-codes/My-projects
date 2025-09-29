import pandas as pd
import glob
import re
import os
from DELPHI.LPdetection import LPPreprocess, LPtrain, LPDetect  # Update to your actual import path
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Output folders for plots
results_dir = os.path.join(project_root, "results")
results_dir = os.path.join(project_root, "results", "combined_t1t2") #store it under baseline_t1 results
kmeans_dir = os.path.join(results_dir, "kmeans_plots")
failed_dir = os.path.join(results_dir, "failed_plots")
os.makedirs(kmeans_dir, exist_ok=True)
os.makedirs(failed_dir, exist_ok=True)

LPPreprocess(
    impath=os.path.join(project_root, "data_6/combined/images"),
    lppath=os.path.join(project_root, "data_6/combined/annotations"),
    train_ratio = 0.8, 
    seed=50,
    output_dir=os.path.join(project_root, "data_6/combined")
)

train_counts = [1,2,3,4,5,6,7,8,9,10,13,16,19,22,25,27,30]
results = []

train_impath='./data_6/combined/train_images'
train_lppath='./data_6/combined/train_annotations'

test_impath='./data_6/combined/test_images'
test_lppath='./data_6/combined/test_annotations'

for n in train_counts:

    #### TRAINING 
    trainer = LPtrain(impath = train_impath, lppath = train_lppath, num_images = n)
    combined_mask = trainer.spatial_learning(lpradius=25, plot = False)
    
    # Save interactive Plotly figure as .html instead of .png
    plot_path = os.path.join(kmeans_dir, f"t1_kmeans_iter_{n}.html")
    
    #execute the function
    threshold, cluster_gamma, S_gamma, fig = trainer.color_learning(lpradius=3, backgroundradius=25, k=7, plot = True) #plot all 17 iterations of k-means clustering
    
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
    detector = LPDetect(impath = test_impath, lppath = test_lppath, num_images = None, threshold = threshold, cluster_gamma = cluster_gamma, S_gamma = S_gamma)
    detector.gray_value_image(plot=False)
    detector.binary_mask_image(plot=False)
    detector.apply_train_mask(combined_mask, plot=False)
    detector.connect_and_weight_regions(plot=False)
    detector.predict_laserpoint(trainer.image_dict)
    #detector.plot_predictions()
    precision, recall, f1, bad_ids = detector.performance_test()
    detector.visualise_failed_predictions(bad_ids, lpradius= 50) #plot all 17 iterations of failed plots clustering

    # Saving the plots under results/baseline_t1/failed_plots
    failed_plot_path = os.path.join(failed_dir, f"t1_failed_iter_{n}.png")
    detector.visualise_failed_predictions(
        bad_ids, lpradius=50, save_path=failed_plot_path)
    print(f"Saved failed prediction plot to {failed_plot_path}")
    
    #storing all results
    results.append({
        "num_train_images": n,
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
os.makedirs(results_dir, exist_ok=True)     # ensure it exists
df_results.to_csv(os.path.join(results_dir, "combined_t1t2.csv"), index=False)
print("Saved to:", os.path.join(results_dir, "combined_t1t2.csv"))



