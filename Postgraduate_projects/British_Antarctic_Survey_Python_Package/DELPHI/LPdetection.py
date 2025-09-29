import cv2 as cv
import sys
import matplotlib.pyplot as plt
import json
import glob
import re
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.ndimage import sum as ndi_sum
from scipy.ndimage import label, center_of_mass
from itertools import combinations, permutations
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
import random
import shutil
from tqdm import tqdm
import math
import pandas as pd
from collections import Counter

class LPPreprocess:
    def __init__(self, impath, lppath, split =  True, train_ratio = 0.8, validation = False, seed=100, output_dir = None):

        """
        Matches image and annotation files by ID, then splits them into training, testing, and optionally validation sets,
        saving the results into structured output directories.

        Given paths to .JPG images and corresponding .json laser point annotations, this class matches files by shared IDs,
        shuffles them, and performs a train/test (and optional validation) split. The matched and split datasets are saved
        into dedicated folders for images and annotations in the specified or default output location.

        Args:
            impath (str): Path to the directory containing image files (.JPG).
            lppath (str): Path to the directory containing laser point annotations (.json).
            split (bool, optional): Whether to automatically split the matched files into train/test sets. Defaults to True.
            train_ratio (float, optional): Proportion of data to use for training. Defaults to 0.8 (i.e., 80% train, 20% test).
            validation (bool, optional): Whether to include a validation set (20% of training data). Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to 100.
            output_dir (str, optional): Path where the output train/test/validation folders should be created.
                                        If None, defaults to the parent directory of `impath`.
                                        
        """
        self.impath = impath
        self.lppath = lppath
        self.matched_files = self.get_matched_files()
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(self.impath, ".."))
        if split: 
            self.split = self.train_test_split_and_copy(output_dir,train_ratio = train_ratio, validation = validation , seed = seed)
        else:
            self.split = None # added a splitting argument
            print("No automatic splitting was done using LPPreprocess, please manually split train/test sets for LPTrain and LPDetect")


    def get_matched_files(self):
        """
        Matches image files and laser point annotation files by shared ID, identifies duplicates and mismatches, and returns valid pairs.

        This method scans the specified image and annotation directories, extracting a four-digit numeric ID from filenames
        following the pattern 'IMG_XXXX.JPG' and 'IMG_XXXX.json'. It identifies:
        - Matched image-annotation pairs (by shared ID)
        - Images without corresponding annotations
        - Annotations without corresponding images
        - Duplicate annotation IDs

        It prints summary statistics and returns a list of matched file pairs.

        Returns:
            matched_files (list of tuples): Each tuple contains (full_image_path, full_annotation_path) for a valid ID match.
        
        """
        image_files = glob.glob(os.path.join(self.impath, "*.JPG")) #path to each image and annotation
        annotation_files = glob.glob(os.path.join(self.lppath, "*.json"))

        #Extracting the first capturing group from the regex match with pattern, the pattern being an argument
        extract_id = lambda f, pattern: re.search(pattern, os.path.basename(f)).group(1) \
            if re.search(pattern, os.path.basename(f)) else None

        #Creating a dictionary where each key is the img_id, and the content being their path
        img_ids = {
            img_id: f
            for f in image_files
            if (img_id := extract_id(f, r'IMG_(\d{4})\.JPG')) is not None
        }
        ann_ids = {
            ann_id: f
            for f in annotation_files
            if (ann_id := extract_id(f, r'IMG_(\d{4})\.json')) is not None
        }
        
        # Checking for duplicate annotation IDs
        ann_id_list = [extract_id(f, r'IMG_(\d{4})\.json') for f in annotation_files]
        ann_counts = Counter(ann_id_list)
        duplicates = [aid for aid, count in ann_counts.items() if count > 1]
        print(f"Duplicated annotations: {duplicates}")
        
        #turning things into a set can help me prepare to find the intersection at the next step
        img_set = set(img_ids) 
        ann_set = set(ann_ids)
        
        #listing all IDs without matches in order
        only_images = sorted(img_set - ann_set) 
        only_annotations = sorted(ann_set - img_set)
        
        matched_ids = sorted(set(img_ids) & set(ann_ids))

        print(f"Found {len(img_ids)} images")
        print(f"Found {len(ann_ids)} annotations")
        print(f"Matched pairs: {len(matched_ids)}")
        print(f"Unmatched images: {only_images}")
        print(f"Unmatched annotations: {only_annotations}")

        # the path of the img/annotation with id i, put into the list of matched_files
        matched_files = [(img_ids[i], ann_ids[i]) for i in matched_ids]

        return matched_files #return matched list
    
    def train_test_split_and_copy(self, output_dir, train_ratio=None, validation = False, seed=None):
        """
        Splits matched image-annotation pairs into train, test, and optional validation sets, then copies them into organized folders.

        The matched pairs are shuffled using a fixed random seed, then split into training and testing sets based on the specified ratio.
        If validation is enabled, 20% of the training data is further set aside for validation. All data are copied into subdirectories
        under the given output folder, such as 'train_images', 'train_annotations', etc.

        Args:
            output_dir (str): Destination path where new train/test/val folders will be created.
            train_ratio (float, optional): Fraction of matched data to assign to the training set. Defaults to 0.7.
            validation (bool, optional): If True, creates a validation set using 20% of the training data. Defaults to False.
            seed (int, optional): Random seed for reproducible shuffling. Defaults to 100.

        Returns:
            dict: A dictionary with keys 'train', 'val', and 'test', each containing a list of 4-digit image IDs used in that split.
        
        """
        train_ratio = 0.7 if train_ratio is None else train_ratio
        seed = 100 if seed is None else seed
        random.seed(seed)
        random.shuffle(self.matched_files)
        split_idx = int(len(self.matched_files) * train_ratio)
        train = self.matched_files[:split_idx]
        test = self.matched_files[split_idx:]
        val = []
        
        if validation:
            val_split_idx = len(train) // 4
            val = train[:val_split_idx]
            train = train[val_split_idx:]
            
        import traceback
        
        # Create folder list
        folders = ['train_images', 'train_annotations', 'test_images', 'test_annotations']
        
        if validation:
            folders += ['val_images', 'val_annotations']

        for folder in folders:
            full_path = os.path.join(output_dir, folder)
            try:
                if os.path.exists(full_path):
                    shutil.rmtree(full_path)
                os.makedirs(full_path, exist_ok=True)

            except Exception as e:
                print(f"[ERROR] Failed to create folder: {full_path}")
                print(f"[EXCEPTION] {e}")
                traceback.print_exc()

#copying the data into images and annotation pairs into the folders
        def copy_pairs(pairs, prefix):
            for img_path, ann_path in pairs:
                try:
                    img_filename = os.path.basename(img_path)
                    ann_filename = os.path.basename(ann_path)

                    img_dest = os.path.join(output_dir, f"{prefix}_images", img_filename)
                    ann_dest = os.path.join(output_dir, f"{prefix}_annotations", ann_filename)

                    shutil.copy(img_path, img_dest)
                    shutil.copy(ann_path, ann_dest)
                except Exception as e:
                    print(f"[ERROR] Failed to copy pair: {img_path}, {ann_path}")
                    print(f"[EXCEPTION] {e}")
                    traceback.print_exc()
    
        copy_pairs(train, "train")
        if validation:
            copy_pairs(val, "val")
        copy_pairs(test, "test")
    
        print(f"Matched {len(self.matched_files)} pairs → {len(train)} train, {len(val)} validation, {len(test)} test.")
        
        def extract_id(path):
            """
            Extracting the ids from the test and train split to be displayed.

            """
            match = re.search(r'IMG_(\d{4})', os.path.basename(path))
            return match.group(1) if match else None

        train_ids = [extract_id(img_path) for img_path, _ in train]
        val_ids = [extract_id(img_path) for img_path, _ in val] if validation else []
        test_ids = [extract_id(img_path) for img_path, _ in test]

        return {"train": train_ids, "val": val_ids, "test": test_ids}

class LPtrain():
    
    def __init__(self, impath, lppath, num_images = None):
        """
        Loads training images and laser point annotations into a dictionary, and provides methods for visualization and feature learning.

        This class constructs a dictionary keyed by image ID, where each entry contains the RGB image and a list of laser point coordinates.
        It supports both full dataset usage and limited subsampling via `num_images`.

        Args:
            impath (str): Path to folder containing .JPG training images.
            lppath (str): Path to folder containing .json laser point annotations.
            num_images (int, optional): Number of image-annotation pairs to include. Defaults to using all matched pairs.
        
        """
        self.impath = impath
        self.lppath = lppath
        if num_images is None: 
            self.num_images = len(glob.glob(f"{impath}/*.JPG"))
        else:
            self.num_images = num_images
        self.image_dict = self.read_and_preprocess()  # Reading images and laser point labels
        
    def read_and_preprocess(self): 
        """
        Reads training images and annotation files, then builds a dictionary indexed by image ID.

        For each matched image-annotation pair, it:
        - Loads the image using OpenCV and converts from BGR to RGB.
        - Reads the laser point JSON file and extracts center points from bounding boxes.
        - Stores the processed image and laser points in a dictionary keyed by 4-digit ID.

        Returns:
            dict: {
                "XXXX": {
                    "image": np.ndarray (H x W x 3),
                    "laser_points": list of (x, y) coordinates
                }
            }
       
        """


        # This pattern should be present in all images and laser point files
        imgpattern = r'.*IMG_(\d{4})\.JPG'  
        lppattern = r'.*IMG_(\d{4})\.json'  

        # Searching through the folder and finding files that are JPG images and JSON laser point annotations
        paths_to_images = glob.glob(f"{self.impath}/*.JPG")  # Find all JPG images

        # Find the JSON files
        paths_to_lp = glob.glob(f"{self.lppath}/*.json")

        # If there are no JSON files found, print error
        if not paths_to_lp:  
            print(os.listdir(self.lppath))  
            sys.exit("Error: No JSON files found in the given path.")
            
        if not paths_to_images:  
            sys.exit("Error: No JPG files found in the given path.")

        extract = lambda path, pattern: re.search(pattern, os.path.basename(path)).group(1) \
            if re.search(pattern, os.path.basename(path)) else None
            
        img_map = {
                extract(p, imgpattern): p
                for p in sorted(paths_to_images)
                if extract(p, imgpattern) is not None
            }
        lp_map = {
            extract(p, lppattern): p
            for p in sorted(paths_to_lp)
            if extract(p, lppattern) is not None
        }
        
        #matched_ids = list(set(img_map) & set(lp_map)) this is changed, because set is unordered/random, which makes it unproducible when doing list(set)
        matched_ids = sorted(set(img_map) & set(lp_map))
        random.seed(100) 
        random.shuffle(matched_ids)
        selected_ids = matched_ids[:self.num_images]
    
        print(f"The selected images for this {'test' if isinstance(self, LPDetect) else 'train'} set are: {selected_ids}")
        
        img_dict = {}
        for img_id in selected_ids:
            # Changing image from BGR (openCV uses this as a default), to RGB
            image_bgr = cv.imread(img_map[img_id])
            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

            # This changes from a bounding box to one (x,y) laser point
            with open(lp_map[img_id], "r") as file:
                lp_data = json.load(file)
                shapes = lp_data.get("shapes", [])
                center_points = []
                for shape in shapes:
                    points = shape.get("points", [])
                    if len(points) == 2:
                        (x1, y1), (x2, y2) = points
                        x_center = int((x1 + x2) // 2)
                        y_center = int((y1 + y2) // 2)
                        center_points.append((x_center, y_center))

            img_dict[img_id] = {
                "image": image_rgb,
                "laser_points": center_points
            }

        return img_dict

    def imgdisplay(self, img_id, lpradius = 25, save_path = None):
        """
        Display the original image, the binary mask of laser points, and print laser point coordinates.

        Args:
            img_id: 4-digit ID of the image to display.
            lpradius: the size of the circle printed for the laser point
            save_path: specify path to save the image to.
        
        """

        # Checking if the image exists
        if img_id not in self.image_dict:
            print(f"Error: Image with ID {img_id} not found.")
            return

        image = self.image_dict[img_id]["image"]

        # check if the image is None (in case of a failed read)
        if image is None:
            print(f"Error: Image with ID {img_id} could not be loaded.")
            return
        
        #make the binary mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        laser_points = self.image_dict[img_id]["laser_points"]
        for (x_center, y_center) in laser_points:
            cv.circle(mask, (int(x_center), int(y_center)), lpradius, 255, -1)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title(f"Image {img_id}")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"Laser Points Mask {img_id}")
        ax[1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

        
        print(f"Laser Point Coordinates for Image {img_id}: {self.image_dict[img_id]['laser_points']}")
    
    def spatial_learning(self, lpradius=25, plot = False):
        """
        Generates binary masks around laser point locations for each training image and stores them.

        Each pixel within the specified radius around a laser point is set to 255 (foreground),
        while all other pixels remain 0 (background). The mask is saved under each image entry in `self.image_dict`.

        Args:
            lpradius (int): Radius (in pixels) to dilate each laser point to include in the mask.
            plot (bool): If True, displays the union of all binary masks for visual debugging.

        Returns:
            np.ndarray: Combined binary mask representing the union of all training masks.
        
        """

        # Extracting shape from the first image
        first_image = next(iter(self.image_dict.values()))["image"]
        height, width = first_image.shape[:2]

        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for img_id in self.image_dict:
            mask = np.zeros((height, width), dtype=np.uint8)
            laser_points = self.image_dict[img_id]["laser_points"]


            for (x_center, y_center) in laser_points:
                cv.circle(mask, (int(x_center), int(y_center)), lpradius, 255, -1)

            # Storing this individual mask in the dictionary
            self.image_dict[img_id]["mask"] = mask

            # Updating the combined mask (master mask)
            combined_mask = np.maximum(combined_mask, mask)
        
        # plotting mask
        if plot:    
            plt.imshow(combined_mask, cmap='gray')
            plt.title("Combined Binary Mask")
            plt.show()

        return combined_mask

    def color_learning(self, lpradius=3, backgroundradius=25, k=7, plot=False):
        """
        Learns the curated laser point colours using k-means clustering and sets a similarity threshold.

        This method extracts RGB pixel values near the laser points (S⁺) and in surrounding background areas (S⁻),
        clusters them using k-means, and identifies the cluster most enriched in laser point pixels.
        It then calculates a Euclidean distance threshold based on intra-cluster spread.

        Args:
            lpradius (int): Radius for collecting laser point pixels (S⁺).
            backgroundradius (int): Radius for collecting background perimeter pixels (S⁻).
            k (int): Number of clusters to use in k-means.
            plot (bool): If True, returns a 3D RGB scatter plot showing clustering.

        Returns:
            float: d_gamma — mean Euclidean distance to the best LP cluster center.
            np.ndarray: c_gamma — RGB centroid of the laser point cluster.
            np.ndarray: s_gamma — RGB pixels assigned to the best LP cluster.
            plotly.Figure or None: 3D RGB visualization if `plot=True`; otherwise None.
        
        """
        S_plus = []   # pixels that are near the LP, within the radius of 3 (see slide 4)
        S_minus = []  # background pixels (see slide 5)
        
        #Get the dimensions of the images 
        first_id = next(iter(self.image_dict))
        first_image = self.image_dict[first_id]["image"]
        height, width = first_image.shape[:2] 
        
        
#LOOP to extract S+ and S- colors from each image using lpradius and lpbackground
        for img_id in self.image_dict: 
            laserpoints = self.image_dict[img_id]["laser_points"]
            image = self.image_dict[img_id]["image"]
            
            #make an empty mask to extract S+
            lp_mask = np.zeros((height, width), dtype=np.uint8) 
            
            #fill the mask with circles around laser points, marking anything in the circle as 255
            for (x, y) in laserpoints:
                cv.circle(lp_mask, (int(x), int(y)), lpradius, 255, -1)
            
            #use the mask to extract pixels from the image
            lp_pixels = image[lp_mask == 255] 
            
            #append to S_plus
            S_plus.extend(lp_pixels) 
            
            #make an empty mask to extract S-, marking anything in the perimeter of the circle as 255
            background_mask = np.zeros((height, width), dtype=np.uint8)
            for (x, y) in laserpoints: 
                cv.circle(background_mask, (int(x), int(y)), backgroundradius, 255, 1) 
            
            #use the mask to extract the permiter values
            background_pixels = image[background_mask == 255] 
            
            #append to S_minus
            S_minus.extend(background_pixels)
            
    #Combine S+ and S- pixels (see slide 6) and do k-means clustering

        # Combining the sets 
        S = np.array(S_plus + S_minus)

        # kmeans clustering 
        kmeans = KMeans(n_clusters=k, random_state=100, n_init='auto') #randomly intiitlaise 7 clusters 
        kmeans.fit(S)
        
        
        labels = kmeans.labels_ #labelling each RGB value to correspond to its cluster center
        #ie. labels = [0, 2, 1, 0, 2, 1, 0, ...] pixel 1 in the S array is cluster 0, pixel 2 in the S array is cluster 2 etc
        
        cluster_centers = kmeans.cluster_centers_ #RGB coordinates of each cluster center
        #ie. contains 7 points, each with RGB of the cluster center 
    
        ratios = []
        N_plus = len(S_plus) 
        for cluster_id in range(k):
            labels_S_plus = labels[:N_plus] #labels that are laser points
            #if it matches the particular cluster id in the loop, add it into S_k_plus_points
            S_k_plus_points = labels_S_plus == cluster_id 
            
            #add together all the points for that cluster id
            S_k_plus = np.sum(S_k_plus_points) 
            
            #total labels in that cluster 
            S_k_points = labels == cluster_id 
            S_k = np.sum(S_k_points) 

            #Calculate a ratio (see slide 8)
            ratio = S_k_plus / S_k if S_k > 0 else 0.0
            ratios.append(ratio) #calculate ratio.
                
        #find the highest ratio, this is the most likely to be most representative cluster center (See slide 9)
        gamma = np.argmax(ratios) #index of the largest ratio in the list
        cluster_gamma = cluster_centers[gamma] #RGB of the center (C_y)
        S_plus_np = np.array(S_plus)
        S_gamma = S_plus_np[labels[:N_plus] == gamma]  #All the RGB points assigned to this cluster that are laser points
        if len(S_gamma) > 0:
            dists = cdist(S_gamma, [cluster_gamma], metric='euclidean')
            threshold = float(dists.mean())
        else:
            raise ValueError(
                f"No LP pixels assigned to cluster {gamma}. "
                "Cannot compute threshold — check LP data or clustering parameters."
            )
            
        if plot:
            import plotly.graph_objects as go

            # Convert to numpy
            S_plus_np = np.array(S_plus)
            S_minus_np = np.array(S_minus)

            # Subsample for clarity
            if len(S_plus_np) > 3000:
                S_plus_np = S_plus_np[np.random.choice(len(S_plus_np), 3000, replace=False)]
            if len(S_minus_np) > 3000:
                S_minus_np = S_minus_np[np.random.choice(len(S_minus_np), 3000, replace=False)]

            fig = go.Figure()

            # Laser Points (S+)
            fig.add_trace(go.Scatter3d(
                x=S_plus_np[:, 0], y=S_plus_np[:, 1], z=S_plus_np[:, 2],
                mode='markers',
                marker=dict(size=3, color='red', opacity=0.5),
                name='Laser Points (S⁺)'
            ))

            # Background Points (S-)
            fig.add_trace(go.Scatter3d(
                x=S_minus_np[:, 0], y=S_minus_np[:, 1], z=S_minus_np[:, 2],
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.2),
                name='Background (S⁻)'
            ))

            # Cluster centers
            for idx, center in enumerate(cluster_centers):
                percent = f"{ratios[idx] * 100:.0f}%"
                is_best = idx == gamma
                label = f"Cluster {idx} ({'Best Cluster, ' if is_best else ''}{percent})"

                fig.add_trace(go.Scatter3d(
                    x=[center[0]], y=[center[1]], z=[center[2]],
                    mode='markers+text',
                    marker=dict(
                        size=8,
                        color='green' if is_best else 'black',
                        symbol='diamond' if is_best else 'circle'
                    ),
                    text=[label],   # Displayed on the plot
                    name=label      # Legend label
                ))



            fig.update_layout(
                title='RGB Color Clustering: S⁺ and S⁻ Pixels',
                scene=dict(
                    xaxis_title='Red',
                    yaxis_title='Green',
                    zaxis_title='Blue'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(x=0.02, y=0.98)
            )

            return threshold, cluster_gamma, S_gamma, fig #d_y, c_y, s_y
        else:
            return threshold, cluster_gamma, S_gamma, None
            
        

class LPDetect(LPtrain):
    def __init__(self, impath, lppath, threshold, cluster_gamma, S_gamma, num_images = None):
        """
        Initialize with parameters learned during training.
        Args:
            threshold (float): d_gamma from training
            cluster_gamma (np.ndarray): RGB centroid of LP cluster
            S_gamma (np.ndarray): RGB vectors of LP pixels in cluster gamma
        
        """
        super().__init__(impath, lppath, num_images) #does the same things with impath, lppath and read_and_preprocess as train __init__ did. 
        self.threshold = threshold
        self.cluster_gamma = cluster_gamma
        self.S_gamma = S_gamma
        
    def plot_image_dict_grid(
        self,
        image_dict,
        key,
        title_prefix="",
        cmap='gray',
        vmin=None,
        vmax=None,
        dilate=True,
        dilation_kernel_size=50,
        save_path = None
    ):
        """
        Generic subplot grid for visualizing 2D images in image_dict[img_id][key].

        Args:
            image_dict (dict): Dictionary containing image data per image ID.
            key (str): Key in the inner dict to plot (e.g., 'gray_value_image', 'binary_image').
            title_prefix (str): Prefix to show before each subplot title.
            cmap (str): Matplotlib colormap to use.
            vmin, vmax: Value range for imshow (for consistent contrast across subplots).
            dilate (bool): Whether to dilate the mask image before plotting (useful for binary masks).
            dilation_kernel_size (int): Size of the square kernel for dilation (default 50x50).
        
        """
        img_ids = list(image_dict.keys())
        num_plots = len(img_ids)
        cols = 3
        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for idx, img_id in enumerate(img_ids):
            ax = axes[idx]
            data = image_dict[img_id].get(key)

            if data is not None:
                if dilate:
                    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
                    data = cv.dilate(data, kernel, iterations=1)

                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title_prefix}{img_id}")
                ax.axis('off')
                fig.colorbar(im, ax=ax, shrink=0.8)

        
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[SAVED] Grid plot saved to {save_path}")
        else:
            plt.show()

# gray value image (see slide 11)
    
    def gray_value_image(self,plot=True,save_path=None):
        """
        Computes a gray-scale confidence map for each image based on RGB similarity to laser point colors.

        Uses a KDTree to compare each image pixel to known LP RGB vectors (`S_gamma`) and maps closeness
        to a confidence score. Bright regions in the resulting image correspond to pixels that resemble LP colors.

        Args:
            plot (bool): If True, plots the gray value images as a grid.
            save_path (str, optional): If provided, saves the plot to this path instead of displaying it.

        Modifies:
            self.image_dict[img_id]["gray_value_image"]: 2D float map of LP color similarity.
        
        """
        tree = cKDTree(self.S_gamma) #sorts my S_gamma values into a decision tree, with each split using median values of R,G,B recursively
        #Until leaf is reached for each point. 
        # (see slide 12)

        for img_id in tqdm(self.image_dict, total=len(self.image_dict)):
            image_rgb = self.image_dict[img_id]["image"]
            H, W, _ = image_rgb.shape
            image_flat = image_rgb.reshape(-1, 3).astype(np.float32)
            min_dists, _ = tree.query(image_flat, k=1)
            
            #--- notes ---
            # I tried this too and it doesnt work dists = np.linalg.norm(image_flat[:, None, :] - self.S_gamma[None, :, :], axis=2) #numpy broadcast, vectorized version
            # instead of dists = cdist(image_flat, self.S_gamma, metric='euclidean') where i am computing distances between all pixels and all LP RGB samples
            # For example for one pixel, I will have calculated the distance of that pixel's RGB to LP 1's color, LP 2's color etc etc.
            #print("For each pixel, compute the shortest distance to any of the known LP RGB colors...")
            #min_dists = np.min(dists, axis=1)  # for each pixel, choose the smallest distance to a LP
            
            #Calculate a gray_value_image, where brighter pixels represent when min_dist value is much smaller than the learnt threshold (more LP like)
            g = np.maximum(0, (self.threshold - min_dists) / (self.threshold**2)) 
            g_map = g.reshape(H, W)  # reshaping this value into 2D image to be plotted
            self.image_dict[img_id]["gray_value_image"] = g_map
            
        if plot:
            self.plot_image_dict_grid(self.image_dict, key = "gray_value_image", title_prefix="Grey image values ", cmap='gray', vmin=None, vmax=None, save_path = save_path)
            
        return None

# (see slide 13)
    def binary_mask_image(self, morphology = True, kernel_dim = 3, plot = True, save_path = None):
        """
        Converts gray value images into binary masks and optionally applies morphological opening.

        Pixels with non-zero confidence are considered potential LPs and labeled 1.
        Morphological filtering (3x3 by default) removes small noisy components.

        Args:
            morphology (bool): Whether to apply morphological opening.
            kernel_dim (int): Size of the square kernel used for morphological filtering.
            plot (bool): If True, shows the binary masks for all images.
            save_path (str, optional): If provided, saves the plot instead of displaying.

        Modifies:
            self.image_dict[img_id]["binary_image"]: Cleaned binary mask from gray values.
        
        """
        for img_id in self.image_dict:
            g_map = self.image_dict[img_id]["gray_value_image"]
            binary_mask = (g_map > 0).astype(np.uint8) 
            print(f"[{img_id}] Raw binary pixels before morphology: {np.count_nonzero(binary_mask)}") 
            if morphology:
                kernel = np.ones((kernel_dim, kernel_dim), np.uint8) 
                binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)  # Clean up noise with morphological opening 3x3 kernel
                print(f"[{img_id}] Binary pixels after morphology: {np.count_nonzero(binary_mask)}")
            self.image_dict[img_id]["binary_image"] = binary_mask
        
        if plot:
            self.plot_image_dict_grid(self.image_dict, key = "binary_image", title_prefix="Morphological opening ", cmap='gray', vmin=None, vmax=None, save_path = save_path)

        return None
    
#NOT mentioned in the paper, but it was included in the code given to me by timm
    def apply_train_mask(self, combined_mask, plot = True, save_path = None):
        """
        Filters the predicted binary mask of each test image using a training mask.

        Multiplies the test image's binary mask by a spatial mask derived from training images,
        suppressing unlikely regions.

        Args:
            combined_mask (np.ndarray): Global binary mask from training images.
            plot (bool): If True, displays the filtered masks.
            save_path (str, optional): If provided, saves the plot instead of displaying.

        Modifies:
            self.image_dict[img_id]["binary_image"]: Mask after applying spatial constraint.
        
        """
        for img_id in self.image_dict:
            test_image_mask = self.image_dict[img_id]["binary_image"]
            spatial_filtered_mask = test_image_mask * combined_mask
            self.image_dict[img_id]["binary_image"] = spatial_filtered_mask
            
        if plot:
            self.plot_image_dict_grid(self.image_dict, key = "binary_image", title_prefix="Master Mask filter ", cmap='gray', vmin=None, vmax=None, save_path = save_path)

        return None
    
### see slide 14

    def connect_and_weight_regions(self, target_number_of_regions = 5, connectivity = 8, plot = True, save_path = None):
        """
        Identifies connected regions in each binary mask and ranks them based on summed gray values.

        For each image, connected regions are labeled and scored by the sum of confidence values (from gray value image).
        The top N regions (default 5) are selected as LP candidates, and their centroids are stored.

        Args:
            target_number_of_regions (int): Number of top regions to keep as LP candidates.
            connectivity (int): Use 4 or 8 for neighborhood connectivity (8 is default).
            plot (bool): If True, displays a mask of selected regions.
            save_path (str, optional): If provided, saves the visualization to this path.

        Modifies:
            self.image_dict[img_id]["LP_candidates"]: List of (x, y) coordinates for top regions.
            self.image_dict[img_id]["LP_candidates_mask"]: Binary mask with LP candidate centroids marked.
        
        """
        
            # Set the connectivity structure
        if connectivity == 4:
            structure = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]])  # 4-connected (default)
        elif connectivity == 8:
            structure = np.ones((3, 3), dtype=np.uint8)  # 8-way connected
        else:
            raise ValueError("connectivity must be either 4 or 8")

        for img_id in self.image_dict:
            binary_mask = self.image_dict[img_id]["binary_image"]

            # Label connected regions (see slide 14)
            labels, num_labels = label(binary_mask, structure=structure) #label the connected regions 
            print(f"[{img_id}] Connected regions found: {num_labels}")
            
            #sum the gray image values (likelihood/brightness) within a single connected region
            g_map = self.image_dict[img_id]["gray_value_image"]

            #get the centroids of each connected regions (see slide 15) and sum
            centroids = center_of_mass(binary_mask, labels, range(1, num_labels + 1)) 
            region_sums = ndi_sum(g_map, labels, index=range(1, num_labels + 1))
            
            #top five sums are chosen (see slide 15)
            region_weights = list(zip(region_sums, centroids)) # summed likelihood becomes a weight score
            region_weights.sort(reverse=True, key=lambda x: x[0]) # Sort by weight in descending order
        
            # --- notes ----
            # The loop was very SLOW, i replaced with the above
            #region_weights = []  # To store (weight, centroid) for each region
            #for i in range(1, num_labels):  # Background is labeled 0, so we skip it
            #    region_mask = (labels == i)  # Get a mask for this region
            #    weight = confidence_map[region_mask].sum()  # Sum the confidence values within this region
            #    region_weights.append((weight, centroids[i]))  # Store the region's total weight and its centroid

            # FLIP because labelme/openCV have a flipped coordinate from numpy/scipy
            def flip_yx_to_xy(points):
                return [(int(x), int(y)) for (y, x) in points]

            unflipped_centroids = [c for _, c in region_weights[:target_number_of_regions]] #i skipped the weight, as it is only used for sorting, i just want the label 
            top_centroids = flip_yx_to_xy(unflipped_centroids)
                        
            # Each centroid c is in the form of (x, y) coordinates (floats),
            # map(int, c) ensures both x and y become integers, e.g. (29.1, 56.9) -> (29, 56)
            print(f"[{img_id}] LP candidates found: {len(top_centroids)}")

            self.image_dict[img_id]["LP_candidates"] = top_centroids # store regions
            
            candidates = self.image_dict[img_id]["LP_candidates"]
            # Assume shape is known (e.g., original image size)
            h, w = self.image_dict[img_id]["gray_value_image"].shape
            mask = np.zeros((h, w), dtype=np.uint8)
            for (x, y) in candidates:
                mask[int(y), int(x)] = 1
            # Then plot this instead of the raw list
            self.image_dict[img_id]["LP_candidates_mask"] = mask
        
        if plot:
            self.plot_image_dict_grid(self.image_dict, key = "LP_candidates_mask", title_prefix="Region selection ", cmap='gray', vmin=None, vmax=None, save_path = save_path)

        return None


    def _triangle_distance(self, candidate, reference):
        """
        Computes the minimum permutation-invariant Euclidean distance between a candidate triangle and a reference triangle.

        Tries all permutations of the candidate triangle points to find the best match to the reference triangle.

        Args:
            candidate (np.ndarray): Shape (3, 2), list of 3 (x, y) points.
            reference (np.ndarray): Shape (3, 2), reference triangle points.

        Returns:
            float: Minimum total distance between permuted candidate and reference triangle.
        
        """
        
        min_cost = float('inf')
        for perm in permutations([0, 1, 2]):
            reordered = candidate[list(perm)]
            cost = np.sum(np.linalg.norm(reordered - reference, axis=1))
            min_cost = min(min_cost, cost)
        return min_cost

# Match up triangles to the training traingles, to find minimum diff (see slide 16-24)
    def predict_laserpoint(self, train_dict):
        """
        Predicts laser point locations by matching candidate triangles to known training triangles using geometric similarity.

        For each test image, this method:
        - Generates all 3-point combinations from region centroids (LP candidates).
        - Computes the permutation-invariant Euclidean distance from each candidate triangle to all training triangles.
        - Selects the triangle with the lowest distance (best geometric match).
        - If fewer than 3 candidates are available, the raw candidates are returned without matching.

        Args:
            train_dict (dict): Dictionary containing training images with 'laser_points' entries (ground truth LP triangles).

        Returns:
            dict: Mapping from image ID to a tuple:
                - best_triangle (list of 3 (x, y) tuples): The predicted laser point triangle.
                - best_cost (float): The sum of Euclidean distances from the best-matched training triangle.
                If fewer than 3 LP candidates are available, returns (raw_candidates, None).
        
        """
        results = {}

        # Build list of training triangles
        training_triangles = [
            np.array(train_dict[img_id]["laser_points"])
            for img_id in train_dict
        ]

        for img_id in self.image_dict:
            candidates = self.image_dict[img_id].get("LP_candidates", [])

            if len(candidates) < 3:
                print(f"[Warning] Not enough LP candidates for {img_id} — storing raw candidates")
                self.image_dict[img_id]["predicted_laser_points"] = candidates
                results[img_id] = (candidates, None)
                continue

            triangle_candidates = list(combinations(candidates, 3))
            best_triangle = None
            best_cost = float('inf')

            for tri in triangle_candidates:
                tri_array = np.array(tri)
                for train_tri in training_triangles:
                    cost = self._triangle_distance(tri_array, train_tri)
                    if cost < best_cost:
                        best_cost = cost
                        best_triangle = tri_array

            results[img_id] = (best_triangle, best_cost)
            self.image_dict[img_id]["predicted_laser_points"] = [tuple(int(v) for v in pt) for pt in best_triangle]

        return results


# Testing the performance!

    def performance_test(self, max_dist=25,required_tp=3):
        """
        Evaluates detection performance by comparing predicted laser points with ground truth annotations.

        A predicted point counts as a True Positive (TP) if it's within `max_dist` pixels of an unmatched ground truth point.
        Calculates per-image precision, recall, and F1-score. Also tracks images with fewer than `required_tp` true positives.

        Args:
            max_dist (int): Distance threshold for a prediction to count as a true positive.
            required_tp (int): Minimum number of TPs required for an image to pass.

        Returns:
            tuple: (mean_precision, mean_recall, mean_f1, bad_ids)
                - mean_precision (float): Average precision across images.
                - mean_recall (float): Average recall across images.
                - mean_f1 (float): Average F1-score.
                - bad_ids (list): Image IDs with fewer than `required_tp` true positives.
        
        """
        precision_list = [] 
        recall_list = []
        f1_list = []
        bad_ids = []

        for img_id in self.image_dict:
            gt_points = self.image_dict[img_id].get("laser_points", []) #ground truths
            pred_points = self.image_dict[img_id].get("predicted_laser_points", []) #predicted points
            print(f"ground truth and predicted points for {img_id} is {gt_points} and {pred_points}")

            if not gt_points and not pred_points:
                print(f"Warning: image {img_id} contains no ground truth and prediction points")
                continue

            gt_points = np.array(gt_points) 
            pred_points = np.array(pred_points)
            
            # if there are no ground truths but there are predicted points, then there are only false positives.
            if len(gt_points) == 0:
                tp = 0
                fp = len(pred_points)
                fn = 0
            elif len(pred_points) == 0: # if there are no predicted points but there should've been, then there are only false negatives
                tp = 0
                fp = 0
                fn = len(gt_points)
            else:
                dists = cdist(gt_points, pred_points) #comparing triangles (see slides 16-20)
                matched_gt = set()
                matched_pred = set()

            #IF the match is smaller than the max distance, then it is a true positive.
                tp = 0
                for i, row in enumerate(dists):
                    for j in np.argsort(row):  # from closest to farthest
                        if row[j] <= max_dist and j not in matched_pred:
                            tp += 1
                            matched_gt.add(i)
                            matched_pred.add(j)
                            break
            #False positive is when it is predicted, but there is actually not a ground truth 
                fp = len(pred_points) - tp
            #False negative is when it is not predicted, but there is actually a ground truth
                fn = len(gt_points) - tp #accounts for if i only have 1 predicted point, and 3 ground truth points. if so, fp will be 2

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            
            # ✳️ Add to bad_ids if fewer than required TP
            if tp < required_tp:
                bad_ids.append(img_id)

        mean_precision = np.mean(precision_list) if precision_list else 0.0
        mean_recall = np.mean(recall_list) if recall_list else 0.0
        mean_f1 = np.mean(f1_list) if f1_list else 0.0

        print(f"Evaluation over {len(precision_list)} images:")
        print(f"  Mean Precision: {mean_precision:.3f}")
        print(f"  Mean Recall:    {mean_recall:.3f}")
        print(f"  Mean F1 Score:  {mean_f1:.3f}")

        return mean_precision, mean_recall, mean_f1, bad_ids
        
    def imgdisplay(self, img_id, lpradius = 25, save_path = None):
        """
        Display the original image, the binary mask of laser points, and print laser point coordinates.

        Args:
            img_id: 4-digit ID of the image to display.
            lpradius: the size of the circle printed for the laser point
        
        """

        # Check if the image exists in the dictionary
        if img_id not in self.image_dict:
            print(f"Error: Image with ID {img_id} not found.")
            return

        image = self.image_dict[img_id]["image"]

        # Check if the image is None (in case of a failed read)
        if image is None:
            print(f"Error: Image with ID {img_id} could not be loaded.")
            return
        
        #make the binary mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        laser_points = self.image_dict[img_id]["laser_points"]
        for (x_center, y_center) in laser_points:
            cv.circle(mask, (int(x_center), int(y_center)), lpradius, 255, -1)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title(f"Image {img_id}")
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"Laser Points Mask {img_id}")
        ax[1].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

        
    def visualize_overlay(self, image, mask, predicted_points, alpha=0.5, lpradius=8, title=None, block_size=60, save_path = None):
        """
        Overlays a square-block visualization of the binary mask on the original image, and plots predicted laser points.

        The masked regions are fully visible while the background is dimmed. Predicted LPs are shown as red circles.
        Useful for assessing where predictions lie relative to filtered candidate regions.

        Args:
            image (np.ndarray): Original RGB image (H x W x 3).
            mask (np.ndarray): Binary mask (H x W), where 255 indicates active regions.
            predicted_points (list of (x, y)): List of predicted LP coordinates.
            alpha (float): Background dimming factor. Higher = darker.
            lpradius (int): Radius of red circles representing LPs.
            title (str, optional): Title for the plot.
            block_size (int): Size of square blocks used for mask visualization.
            save_path (str, optional): If given, saves the plot instead of displaying it.
        
        """
        # Ensuring image is RGB
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        image = image.astype(np.float32) / 255.0  

        # Create dimmed background
        dimmed_image = image * (1 - alpha)

        # Create a copy to overlay squares
        overlay = dimmed_image.copy()

        # Get all square blocks where mask is non-zero
        H, W = mask.shape
        for y in range(0, H, block_size):
            for x in range(0, W, block_size):
                block = mask[y:y+block_size, x:x+block_size]
                if np.any(block):  # if any part of block is active
                    overlay[y:y+block_size, x:x+block_size, :] = image[y:y+block_size, x:x+block_size, :]

        # Clipping overlay to [0,1] and convert to uint8
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)

        # Draw red circles for predicted LPs
        for (x, y) in predicted_points:
            circle = plt.Circle((x, y), radius=lpradius, color='red', fill=False, linewidth=1)
            plt.gca().add_patch(circle)

        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[SAVED] Overlay visualization saved to {save_path}")
        else:
            plt.show()
        
    def visualise_failed_predictions(self, failed_ids, alpha=0.5, lpradius=8, block_size=60, save_path=None):
        """
        Visualizes all test images that failed to detect at least 3 laser points.

        For each failed image:
        - Overlays its binary mask using square blocks.
        - Displays predicted laser points (red circles) and ground truth points (white dots).
        - Highlights alignment (or misalignment) between predictions and true LPs.

        Args:
            failed_ids (list): List of image IDs with fewer than 3 true positives.
            alpha (float): Transparency level for dimmed background.
            lpradius (int): Radius of red circle used for predicted LPs.
            block_size (int): Size of the square blocks used for mask overlay.
            save_path (str, optional): If provided, saves the figure instead of showing it.
        
        """
        print(f"\nThese test images failed to predict three laser points ({len(failed_ids)} total):\n{failed_ids}\n")

        cols = 3
        rows = math.ceil(len(failed_ids) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten()

        for idx, img_id in enumerate(failed_ids):
            ax = axes[idx]
            image = self.image_dict[img_id]["image"]
            mask = self.image_dict[img_id].get("mask", np.zeros(image.shape[:2], dtype=np.uint8))
            pred_points = self.image_dict[img_id].get("predicted_laser_points", [])
            gt_points = self.image_dict[img_id].get("laser_points", [])

            # Ensure RGB
            if image.ndim == 2 or image.shape[2] == 1:
                image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            image = image.astype(np.float32) / 255.0
            dimmed = image * (1 - alpha)
            overlay = dimmed.copy()

            # Overlay square mask blocks
            H, W = mask.shape
            for y in range(0, H, block_size):
                for x in range(0, W, block_size):
                    block = mask[y:y+block_size, x:x+block_size]
                    if np.any(block):
                        overlay[y:y+block_size, x:x+block_size, :] = image[y:y+block_size, x:x+block_size, :]

            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
            ax.imshow(overlay)

            # Ground truth LPs (blue, filled dots) — draw first
            if gt_points:
                gt_x, gt_y = zip(*gt_points)
                # Ground truth
                ax.scatter(gt_x, gt_y, color='white', s=30, marker='o', label='GT', zorder=2)

            # Predicted LPs (red, unfilled) — draw after so it's on top
            for (x, y) in pred_points:
                       # Predicted (on top)
                circle = plt.Circle((x, y), radius=lpradius, color='red', fill=False, linewidth=1, zorder=3)
                ax.add_patch(circle)

            ax.set_title(f"Image {img_id} | GT: {len(gt_points)} | Pred: {len(pred_points)}")
            ax.axis('off')

        # Hide unused subplots
        for j in range(len(failed_ids), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Saved failed prediction plot to {save_path}")
            plt.close()
        else:
            plt.show()