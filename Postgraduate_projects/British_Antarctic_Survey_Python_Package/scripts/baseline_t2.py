import pandas as pd
import glob
import re
import os
from DELPHI.LPdetection import LPPreprocess, LPtrain, LPDetect  # Update to your actual import path
import numpy as np
from scipy.spatial.distance import cdist

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Output folders for plots
results_dir = os.path.join(project_root, "results")
results_dir = os.path.join(project_root, "results", "baseline_t2") #store it under baseline_t1 results
kmeans_dir = os.path.join(results_dir, "kmeans_plots")
failed_dir = os.path.join(results_dir, "failed_plots")
os.makedirs(kmeans_dir, exist_ok=True)
os.makedirs(failed_dir, exist_ok=True)

success_counts = {} 

LPPreprocess(
    impath=os.path.join(project_root, "data_1/t2/images"),
    lppath=os.path.join(project_root, "data_1/t2/annotations"),
    train_ratio = 0.8, 
    seed=50,
    output_dir=os.path.join(project_root, "data_1/t2")
)

train_counts = [1,2,3,4,5,6,7,8,9,10,13,16,19,22,25,27,30] #test run
total_runs = len(train_counts)
results = []

train_impath='./data_1/t2/train_images'
train_lppath='./data_1/t2/train_annotations'

test_impath='./data_1/t2/test_images'
test_lppath='./data_1/t2/test_annotations'

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
    
    for img_id in detector.image_dict:
        gt = detector.image_dict[img_id]["laser_points"]
        pred = detector.image_dict[img_id]["predicted_laser_points"]
        if len(gt) == 3 and len(pred) == 3:
            dists = cdist(np.array(gt), np.array(pred))
            match_count = 0
            used_preds = set()
            for i, row in enumerate(dists):
                for j in np.argsort(row):
                    if row[j] <= 25 and j not in used_preds:
                        match_count += 1
                        used_preds.add(j)
                        break
            if match_count == 3:
                success_counts[img_id] = success_counts.get(img_id, 0) + 1
                    
    
    detector.visualise_failed_predictions(bad_ids, lpradius= 50) #plot all 17 iterations of failed plots clustering

    # Saving the plots under results/baseline_t1/failed_plots
    failed_plot_path = os.path.join(failed_dir, f"t2_failed_iter_{n}.png")
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
    
detector.visualise_consistently_successful_predictions(
    success_counts=success_counts,
    min_successes=3,
    top_k=6,
    total_runs=len(train_counts),  # e.g. 17
    lpradius=50,
    save_path=os.path.join(results_dir, "t1_successful_top6.png")
)
    
df_results = pd.DataFrame(results)
os.makedirs(results_dir, exist_ok=True)     # ensure it exists
df_results.to_csv(os.path.join(results_dir, "baseline_t2.csv"), index=False)
print("Saved to:", os.path.join(results_dir, "baseline_t2.csv"))

