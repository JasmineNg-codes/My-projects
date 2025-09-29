import pandas as pd
import os
import shutil
import numpy as np
from sklearn.model_selection import KFold
from DELPHI.LPdetection import LPPreprocess, LPtrain, LPDetect  # Update to your actual import path

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Load all matched image-annotation pairs
preprocessor = LPPreprocess(
    impath=os.path.join(project_root, "data_3/t1/images"),
    lppath=os.path.join(project_root, "data_3/t1/annotations"),
    output_dir=None  # no splitting here
)
all_pairs = preprocessor.matched_files

# Set number of folds
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=100)

# Prepare to store results
results = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

# Perform k-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(all_pairs)):
    print(f"Processing fold {fold+1}/{k_folds}...")

    # Temporary directory for each fold
    temp_dir = os.path.join(project_root, "data_3/t1/cv_split")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    train_pairs = [all_pairs[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]

    # Create folder structure
    for subdir in ["train_images", "train_annotations", "test_images", "test_annotations"]:
        os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)

    # Copy training data
    for img, ann in train_pairs:
        shutil.copy(img, os.path.join(temp_dir, "train_images", os.path.basename(img)))
        shutil.copy(ann, os.path.join(temp_dir, "train_annotations", os.path.basename(ann)))

    # Copy testing data
    for img, ann in test_pairs:
        shutil.copy(img, os.path.join(temp_dir, "test_images", os.path.basename(img)))
        shutil.copy(ann, os.path.join(temp_dir, "test_annotations", os.path.basename(ann)))

    # Train model
    trainer = LPtrain(
        impath=os.path.join(temp_dir, "train_images"),
        lppath=os.path.join(temp_dir, "train_annotations"),
        num_images=None
    )
    combined_mask = trainer.spatial_learning(lpradius=25, plot=False)
    threshold, cluster_gamma, S_gamma, _ = trainer.color_learning(lpradius=3, backgroundradius=25, k=7)

    # Test model
    detector = LPDetect(
        impath=os.path.join(temp_dir, "test_images"),
        lppath=os.path.join(temp_dir, "test_annotations"),
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

    # Store results
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1s.append(f1)

# Aggregate and save results
results.append({
    "num_train_images": len(all_pairs),
    "mean_precision": np.mean(fold_precisions),
    "std_precision": np.std(fold_precisions),
    "mean_recall": np.mean(fold_recalls),
    "std_recall": np.std(fold_recalls),
    "mean_f1_score": np.mean(fold_f1s),
    "std_f1_score": np.std(fold_f1s)
})

results_dir = os.path.join(project_root, "results","kfold_t1")
os.makedirs(results_dir, exist_ok=True)
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(results_dir, "kfold_t1.csv"), index=False)

print("Saved to:", os.path.join(results_dir, "kfold_t1.csv"))
