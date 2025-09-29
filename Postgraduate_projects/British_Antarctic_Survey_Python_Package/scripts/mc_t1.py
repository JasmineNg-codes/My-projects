import pandas as pd
import os
import numpy as np
from DELPHI.LPdetection import LPPreprocess, LPtrain, LPDetect  # Update to your actual import path

script_dir = os.path.dirname(os.path.abspath(__file__))        # e.g. /home/jn492/jn492/scripts
project_root = os.path.abspath(os.path.join(script_dir, "..")) # e.g. /home/jn492/jn492

train_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19, 22, 25, 27, 30]
num_seeds = 5  # number of different seeds to try
results = []

for n in train_counts:
    print(f"Processing {n} training images with {num_seeds} seeds...")

    precisions = []
    recalls = []
    f1_scores = []

    for seed in range(num_seeds):
        # Re-split the data with a new seed
        LPPreprocess(
            impath=os.path.join(project_root, "data_2/t1/images"),
            lppath=os.path.join(project_root, "data_2/t1/annotations"),
            train_ratio = 0.8, 
            seed=seed,
            output_dir=os.path.join(project_root, "data_2/t1")
        )

        train_impath = os.path.join(project_root, "data_2/t1/train_images")
        train_lppath = os.path.join(project_root, "data_2/t1/train_annotations")
        test_impath = os.path.join(project_root, "data_2/t1/test_images")
        test_lppath = os.path.join(project_root, "data_2/t1/test_annotations")

        #### TRAINING
        trainer = LPtrain(impath=train_impath, lppath=train_lppath, num_images=n)
        combined_mask = trainer.spatial_learning(lpradius=25, plot=False)
        threshold, cluster_gamma, S_gamma, _ = trainer.color_learning(lpradius=3, backgroundradius=25, k=7)

        #### TESTING
        detector = LPDetect(
            impath=test_impath,
            lppath=test_lppath,
            num_images=None,
            threshold=threshold,
            cluster_gamma=cluster_gamma,
            S_gamma=S_gamma
        )
        detector.gray_value_image(plot=False)
        detector.binary_mask_image(plot=False)
        detector.apply_train_mask(combined_mask, plot=False)
        detector.connect_and_weight_regions(plot=False)
        detector.predict_laserpoint(trainer.image_dict)
        precision, recall, f1, _ = detector.performance_test()

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"  Seed {seed}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

    # Store mean and std after all seeds
    results.append({
        "num_train_images": n,
        "mean_precision": np.mean(precisions),
        "std_precision": np.std(precisions),
        "mean_recall": np.mean(recalls),
        "std_recall": np.std(recalls),
        "mean_f1_score": np.mean(f1_scores),
        "std_f1_score": np.std(f1_scores)
    })

# Save final results
df_results = pd.DataFrame(results)
results_dir = os.path.abspath(os.path.join(project_root, "results", "mc_t1"))
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, "mc_t1.csv")
df_results.to_csv(csv_path, index=False)
print("Saved to:", csv_path)

