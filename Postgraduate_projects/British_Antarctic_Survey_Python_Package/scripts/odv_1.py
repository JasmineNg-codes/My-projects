import pandas as pd
import numpy as np
import os
import re
from DELPHI.LPdetection import LPtrain

# ---- Paths ----
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

depth_path = os.path.join(project_root, "additional_information", "ps118_images_long_lat_depths.csv")
slope_path = os.path.join(project_root, "additional_information", "ps118_images_slope.csv")

impath_soft = os.path.join(project_root, "data", "soft", "images")
lppath_soft = os.path.join(project_root, "data", "soft", "annotations")

impath_hard = os.path.join(project_root, "data", "hard", "images")
lppath_hard = os.path.join(project_root, "data", "hard", "annotations")

# ---- Load Metadata ----
depth_df = pd.read_csv(depth_path)
slope_df = pd.read_csv(slope_path)

# Extract 4-digit image IDs
depth_df["img_id"] = depth_df["Filename"].str.extract(r"IMG_(\d{4})")
slope_df["img_id"] = slope_df["Filename"].str.extract(r"IMG_(\d{4})")

# Map image ID to metadata
depth_map = depth_df.set_index("img_id")[["Latitude", "Longitude", "Depth (m)"]].to_dict(orient="index")
slope_map = slope_df.set_index("img_id")["slope"].to_dict()

# ---- Function to Extract Info from Trainer ----
def collect_image_metadata(image_dict, substrate_type):
    rows = []
    for img_id, data in image_dict.items():
        laser_points = data.get("laser_points")
        if laser_points is None or len(laser_points) != 3:
            continue
        
        a = np.linalg.norm(np.array(laser_points[0]) - np.array(laser_points[1]))
        b = np.linalg.norm(np.array(laser_points[1]) - np.array(laser_points[2]))
        c = np.linalg.norm(np.array(laser_points[2]) - np.array(laser_points[0]))
        avg_dist = (a + b + c) / 3

        key = img_id[-4:]
        if key in depth_map:
            row = {
                "ID": img_id,
                "Latitude": depth_map[key]["Latitude"],
                "Longitude": depth_map[key]["Longitude"],
                "Depth": depth_map[key]["Depth (m)"],
                "Slope": slope_map.get(key, np.nan),
                "Avg_Laser_Dist": avg_dist,
                "Substrate": f"{substrate_type}"
            }
            rows.append(row)
    return rows

# ---- Run Trainers ----
print("Training soft substrate...")
soft_trainer = LPtrain(impath=impath_soft, lppath=lppath_soft)
soft_data = collect_image_metadata(soft_trainer.image_dict, "soft")

print("Training hard substrate...")
hard_trainer = LPtrain(impath=impath_hard, lppath=lppath_hard)
hard_data = collect_image_metadata(hard_trainer.image_dict, "hard")

# ---- Combine and Save ----
df_combined = pd.DataFrame(soft_data + hard_data)
output_dir = os.path.join(project_root, "additional_information")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "laser_metadata_summary.csv")
df_combined.to_csv(output_path, index=False)

print(f"CSV saved to: {output_path}")
