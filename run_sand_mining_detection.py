
#!/usr/bin/env python3
"""
==========================================================
SAND MINING DETECTION AND MAPPING TOOL v1.4 (Consolidated)
==========================================================
This tool helps detect and map potential sand mining activities
along river systems using satellite imagery and machine learning.

Combines functionalities for:
1. Downloading river-focused images for training.
2. Labeling training images via a GUI.
3. Training a sand mining detection model.
4. Generating probability maps along entire rivers.
5. Creating interactive visualizations and basic reports.

Usage Examples:
  # Train a model using 30 samples along the river
  python run_sand_mining_tool.py --mode train --shapefile path/to/your/river.shp --sample-size 30

  # Create a probability map using a pre-trained model, sampling every 0.5km
  python run_sand_mining_tool.py --mode map --shapefile path/to/your/river.shp --model sand_mining_detection/sand_mining_model.joblib --distance 0.5

  # Train a model and then immediately create a map
  python run_sand_mining_tool.py --mode both --shapefile path/to/your/river.shp --sample-size 25 --distance 1.0
"""

import ee
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import requests
import io
import time
import shutil
import geopandas as gpd
# Import specific shapely types needed
from shapely.geometry import mapping, Point, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
import shapely.geometry # Import the module itself for isinstance checks
from shapely.ops import transform, unary_union
import json
import argparse
from datetime import datetime, timedelta # Make sure datetime is imported
import joblib  # For saving/loading model
# sklearn is imported later in main() for version check if needed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.feature import graycomatrix, graycoprops # Example for texture features
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) # Ignore geopandas future warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignore UserWarnings (e.g., from sklearn/joblib)
# Suppress the specific RequestsDependencyWarning if desired (optional)
warnings.filterwarnings('ignore', category=requests.RequestsDependencyWarning)


# --- Configuration ---
BASE_FOLDER = 'sand_mining_detection'
RIVER_IMAGES_FOLDER = os.path.join(BASE_FOLDER, 'river_images_for_training')
LABELS_FILE = os.path.join(BASE_FOLDER, 'training_labels.csv')
MODEL_FILE = os.path.join(BASE_FOLDER, 'sand_mining_model.joblib')
SCALER_FILE = os.path.join(BASE_FOLDER, 'feature_scaler.joblib')
PROBABILITY_MAPS_FOLDER = os.path.join(BASE_FOLDER, 'probability_maps')
TEMP_FOLDER = os.path.join(BASE_FOLDER, 'temp_processing')
DEFAULT_IMAGE_DIM = 512 # Dimension for downloaded training images
DEFAULT_MAP_IMAGE_DIM = 800 # Dimension for images used in mapping
DEFAULT_BUFFER_METERS = 1500 # Area around points for image download


# --- Earth Engine Initialization ---
def initialize_ee():
    """Initializes Earth Engine with robust error handling and authentication."""
    try:
        # Try standard initialize first, assumes default credentials (gcloud ADC)
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("Successfully initialized Earth Engine (High Volume endpoint - Attempt 1)")
        return True
    except Exception as e:
        # Check if the error indicates already initialized
        if 'already initialized' in str(e).lower():
            print("Earth Engine already initialized.")
            return True
        print(f"Standard EE Initialization failed: {e}. Trying authentication flow...")
        try:
            print("\nAttempting Earth Engine Authentication via gcloud...")
            ee.Authenticate(auth_mode='gcloud') # Use 'gcloud' or 'localhost'
            print("Authentication successful!")
            # Re-initialize after authentication
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            print("Successfully initialized Earth Engine after authentication.")
            return True
        except Exception as auth_e:
            print(f"\nAuthentication or Initialization failed: {str(auth_e)}")
            print("\nPlease ensure:")
            print("1. You have signed up for Google Earth Engine: https://signup.earthengine.google.com/")
            print("2. You have a Google Cloud Project linked to your EE account.")
            print("3. The Earth Engine API is enabled in your Google Cloud Project.")
            print("4. You have the Google Cloud SDK installed and configured (`gcloud auth application-default login`).")
            # Fallback to trying project ID manually if authentication flow failed
            try:
                project_id = input("\nEnter your Google Cloud Project ID (required if auto-detection failed): ")
                if project_id:
                     # Initialize specifying the project
                     ee.Initialize(project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')
                     print(f"Successfully initialized Earth Engine with Project ID: {project_id}")
                     return True
                else:
                    print("Project ID is required for manual initialization.")
                    return False
            except Exception as final_e:
                 print(f"Final initialization attempt with Project ID failed: {final_e}")
                 return False

# --- Sand Mining Detection Class (Training Logic) ---
class SandMiningDetection:
    def __init__(self):
        self.ee_initialized = initialize_ee()
        if not self.ee_initialized:
             raise Exception("Earth Engine could not be initialized. Exiting.")

        self.base_folder = BASE_FOLDER
        self.images_folder = RIVER_IMAGES_FOLDER
        self.labels_file = LABELS_FILE
        self.model_file = MODEL_FILE
        self.scaler_file = SCALER_FILE

        os.makedirs(self.base_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        print(f"Created base folder: {self.base_folder}")
        print(f"Training images will be stored in: {self.images_folder}")

    def load_shapefile_and_get_points(self, shapefile_path, sample_size=20, distance_km=5):
        """Load river shapefile and generate random sample points along it for training."""
        print(f"Loading river shapefile from: {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            # Ensure CRS is geographic (WGS84) for distance calculations
            if gdf.crs is None:
                print("Warning: Shapefile has no CRS defined. Assuming WGS84 (EPSG:4326).")
                gdf.crs = "EPSG:4326"
            elif gdf.crs.to_epsg() != 4326:
                print(f"Converting CRS from {gdf.crs.to_string()} to WGS84 (EPSG:4326)")
                gdf = gdf.to_crs("EPSG:4326")

            # Combine all geometries into a single LineString or MultiLineString
            # Handle Polygons by taking their boundary
            geoms = []
            invalid_geom_count = 0
            for geom in gdf.geometry:
                # Ensure geometry is valid before processing
                if geom is not None and geom.is_valid:
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        if not geom.is_empty: geoms.append(geom.boundary)
                    elif isinstance(geom, (LineString, MultiLineString)):
                        if not geom.is_empty: geoms.append(geom)
                elif geom is not None:
                     invalid_geom_count += 1
                     # print(f"Warning: Invalid geometry found, attempting to buffer(0): {geom.geom_type}") # Reduce noise
                     try:
                          valid_geom = geom.buffer(0)
                          if valid_geom.is_valid and not valid_geom.is_empty:
                               if isinstance(valid_geom, (Polygon, MultiPolygon)):
                                   geoms.append(valid_geom.boundary)
                               elif isinstance(valid_geom, (LineString, MultiLineString)):
                                   geoms.append(valid_geom)
                     except Exception as buffer_err:
                          print(f"  Could not fix geometry with buffer(0): {buffer_err}")

            if invalid_geom_count > 0:
                print(f"Warning: Found and attempted to fix {invalid_geom_count} invalid geometries.")

            if not geoms:
                print("Error: No valid LineString or Polygon geometries found in shapefile after cleaning.")
                return []

            river_geom = unary_union(geoms)
            if river_geom.is_empty:
                 print("Error: Combined river geometry is empty.")
                 return []

            print(f"Total river geometry type: {river_geom.geom_type}")

            points = []
            total_length = river_geom.length # Length in degrees
            approx_length_km = total_length * 111 # Rough estimate

            print(f"Approximate river length: {approx_length_km:.2f} km")
            print(f"Generating {sample_size} random sample points for training...")

            # Generate random points along the line
            if isinstance(river_geom, (LineString, MultiLineString)):
                 min_dist_degrees = (distance_km / 111.0) * 0.5 # Min separation distance

                 attempts = 0
                 max_attempts = sample_size * 100 # Increase attempts further if needed

                 # Create a list of existing point geometries for distance checking
                 existing_point_geoms = []

                 while len(points) < sample_size and attempts < max_attempts:
                    random_fraction = np.random.rand()
                    # Ensure interpolation happens correctly on MultiLineString
                    try:
                        if isinstance(river_geom, MultiLineString):
                            # Select a line segment proportionally to its length (optional, complex)
                            # Simpler: just interpolate along the whole combined length
                            point_geom = river_geom.interpolate(random_fraction, normalized=True)
                        else: # LineString
                            point_geom = river_geom.interpolate(random_fraction, normalized=True)
                    except Exception as interp_err:
                         print(f"Warning: Error during interpolation: {interp_err}")
                         attempts += 1
                         continue # Skip this attempt if interpolation fails

                    # Check minimum distance from existing points
                    is_far_enough = True
                    if existing_point_geoms: # Check if the list is not empty
                        try:
                             # Ensure point_geom is a valid Point before distance calc
                             if isinstance(point_geom, Point):
                                  min_d = min(point_geom.distance(p) for p in existing_point_geoms)
                                  if min_d < min_dist_degrees:
                                      is_far_enough = False
                             else:
                                  # Interpolation might return None or other types on failure/edge cases
                                  is_far_enough = False
                        except Exception as dist_err:
                             print(f"Warning: Error calculating distance: {dist_err}")
                             is_far_enough = False # Skip if distance calc fails

                    if is_far_enough and isinstance(point_geom, Point):
                        points.append((point_geom.y, point_geom.x)) # Store coordinates directly
                        existing_point_geoms.append(point_geom) # Store geometry for next check
                    attempts += 1

                 if len(points) < sample_size:
                     print(f"Warning: Could only generate {len(points)} points with minimum separation. Consider reducing distance_km or sample_size, or check input geometry.")

            else:
                 print("Warning: Combined geometry is not LineString or MultiLineString. Using centroids.")
                 # Fallback: use centroids if line extraction failed
                 point_geoms = [g.centroid for g in gdf.geometry if g is not None and g.is_valid and not g.is_empty]
                 if len(point_geoms) > sample_size:
                      # Use np.random.choice on indices then retrieve geometries/coords
                      chosen_indices = np.random.choice(len(point_geoms), sample_size, replace=False)
                      points = [(point_geoms[i].y, point_geoms[i].x) for i in chosen_indices]
                 else:
                      points = [(pg.y, pg.x) for pg in point_geoms]


            coordinates = points # Already in (lat, lon) format

            print(f"Generated {len(coordinates)} training sample coordinates.")
            return coordinates

        except Exception as e:
            print(f"Error loading shapefile or generating points: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return []

    def download_training_images(self, coordinates, img_dim=DEFAULT_IMAGE_DIM, buffer_m=DEFAULT_BUFFER_METERS):
        """Download Sentinel-2 images for the specified coordinates."""
        if not self.ee_initialized:
             print("Error: Earth Engine not initialized.")
             return False
        if not coordinates:
             print("Error: No coordinates provided for image download.")
             return False

        print(f"\nDownloading {len(coordinates)} images for training/labeling...")
        print(f"Image dimensions: {img_dim}x{img_dim}, Buffer: {buffer_m}m")

        success_count = 0
        # Get current date correctly using datetime
        current_ee_date = ee.Date(datetime.now()) # Get current date once
        start_ee_date = current_ee_date.advance(-12, 'month') # Calculate start date once

        with tqdm(total=len(coordinates), desc="Downloading Images", unit="image") as pbar:
            for i, (lat, lon) in enumerate(coordinates):
                point_start_time = time.time()
                try:
                    point = ee.Geometry.Point([lon, lat])
                    region = point.buffer(buffer_m)

                    # Find the least cloudy Sentinel-2 image in the last 12 months
                    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                        .filterBounds(region) \
                        .filterDate(start_ee_date, current_ee_date) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)) \
                        .sort('CLOUDY_PIXEL_PERCENTAGE')

                    # Use getInfo() cautiously; prefer evaluate for non-blocking size check if needed
                    # For simplicity here, getInfo() on size is acceptable for moderate numbers
                    image_count = s2_collection.size().getInfo()

                    best_image = None # Initialize best_image
                    if image_count > 0:
                         best_image = ee.Image(s2_collection.first()) # Cast to ee.Image
                    else:
                        # Try wider cloud tolerance if needed
                        s2_collection_wider = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                            .filterBounds(region) \
                            .filterDate(start_ee_date, current_ee_date) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
                            .sort('CLOUDY_PIXEL_PERCENTAGE')
                        image_count_wider = s2_collection_wider.size().getInfo() # Re-check count
                        if image_count_wider > 0:
                             best_image = ee.Image(s2_collection_wider.first()) # Cast to ee.Image
                        else:
                           # tqdm.write(f"Skipping point {i+1} ({lat:.4f}, {lon:.4f}): No suitable S2 image found.")
                           pbar.update(1)
                           continue # Skip to next point

                    # Ensure best_image is actually an ee.Image object before proceeding
                    if not isinstance(best_image, ee.Image):
                         # tqdm.write(f"Skipping point {i+1} ({lat:.4f}, {lon:.4f}): Failed to retrieve valid image object.")
                         pbar.update(1)
                         continue

                    # Define visualization parameters (True Color)
                    vis_params = {
                        'bands': ['B4', 'B3', 'B2'], # RGB
                        'min': 0,
                        'max': 3000, # Adjust based on expected reflectance values
                        'gamma': 1.4
                    }

                    # Get the download URL
                    region_coords = region.getInfo()['coordinates']
                    download_url = best_image.getThumbURL({
                        **vis_params, # Unpack vis_params dictionary
                        'region': region_coords, # Pass coordinates directly
                        'dimensions': img_dim,
                        'format': 'png'
                    })

                    # Download the image
                    response = requests.get(download_url, timeout=90) # Increase timeout further
                    response.raise_for_status() # Raise an error for bad status codes

                    # Open image with PIL
                    img = Image.open(io.BytesIO(response.content)).convert('RGB')

                    # Enhance image quality
                    img_enhanced = self.enhance_image(img)

                    # Save the enhanced image
                    filename = f'train_image_{i+1}_{lat:.6f}_{lon:.6f}.png'
                    filepath = os.path.join(self.images_folder, filename)
                    img_enhanced.save(filepath, format='PNG', optimize=True, quality=95)
                    success_count += 1

                    # Dynamically adjust sleep based on request time to aim for ~1 request/sec average
                    elapsed_time = time.time() - point_start_time
                    sleep_time = max(0, 1.2 - elapsed_time) # Aim slightly slower than 1 req/sec
                    time.sleep(sleep_time)


                except ee.EEException as ee_err:
                     ee_err_str = str(ee_err).lower()
                     # Only log unexpected EE errors
                     if not any(term in ee_err_str for term in ['quota', 'rate limit', 'timed out', 'backend error', 'memory limit', 'too many requests']):
                         tqdm.write(f"  EE Error (Point {i+1}): {ee_err_str[:150]}...") # Truncate long messages
                     # Implement exponential backoff for common EE quota/limit errors
                     if any(term in ee_err_str for term in ['quota', 'rate limit', 'too many requests', 'user memory limit', 'computation timed out', 'backend error']):
                         wait_time = min(64, 2**(i % 6 + 1)) # Wait 2, 4, 8, 16, 32, 64s capped
                         tqdm.write(f"    EE Backoff: Waiting {wait_time}s...")
                         time.sleep(wait_time)
                     else:
                         time.sleep(2) # Generic short wait for other EE errors
                except requests.exceptions.RequestException as req_err:
                     tqdm.write(f"  Download Error (Point {i+1}): {req_err}")
                     time.sleep(5) # Wait a bit longer after download errors
                except Exception as e:
                     tqdm.write(f"  Unexpected Error (Point {i+1}): {type(e).__name__} - {e}")
                     import traceback
                     traceback.print_exc() # Print full traceback for unexpected errors
                     time.sleep(2) # General short wait
                finally:
                     pbar.update(1) # Ensure progress bar updates even on error/skip

        print(f"\nSuccessfully downloaded {success_count} out of {len(coordinates)} images.")
        # Return True only if *some* images were downloaded, False otherwise
        return success_count > 0

    def enhance_image(self, img):
        """Apply enhancements to improve image clarity."""
        try:
            # Subtle blur to reduce noise
            img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
            # Unsharp mask for detail
            img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            # Enhance color saturation
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
            # Adjust brightness slightly
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
            # Final sharpen
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            return img
        except Exception as e:
            print(f"  Warning: Error during image enhancement: {e}. Returning original image.")
            return img # Return original if enhancement fails

    # --- Image Labeling GUI ---
    class _LabelingGUI:
        def __init__(self, parent_detector):
            self.detector = parent_detector
            self.images_folder = self.detector.images_folder
            self.labels_file = self.detector.labels_file

            try:
                self.image_files = sorted([f for f in os.listdir(self.images_folder) if f.lower().endswith('.png')])
            except FileNotFoundError:
                messagebox.showerror("Error", f"Training image folder not found:\n{self.images_folder}")
                self.root = None
                return

            if not self.image_files:
                messagebox.showerror("Error", f"No images (.png) found in the training folder:\n{self.images_folder}\nCannot start labeling.")
                self.root = None # Flag that GUI couldn't start
                return

            self.current_index = 0
            self.labels = self._load_labels()

            self.root = tk.Tk()
            self.root.title("Sand Mining Image Labeling Tool")
            self.root.geometry("1000x800") # Adjust size as needed

            # Top Frame: Info and Progress
            self.info_frame = tk.Frame(self.root, pady=10)
            self.info_frame.pack(fill="x")
            self.progress_label = tk.Label(self.info_frame, text="", font=("Arial", 14))
            self.progress_label.pack()

            # Middle Frame: Image Canvas
            self.canvas_frame = tk.Frame(self.root)
            self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
            self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
            self.canvas.pack(fill="both", expand=True)
            # Add binding to resize image when canvas size changes
            self.canvas.bind("<Configure>", self.on_canvas_resize)
            self.current_image_object = None # Store the loaded Image object
            self._image_id_on_canvas = None # Store canvas image ID


            # Bottom Frame: Buttons
            self.button_frame = tk.Frame(self.root, pady=10)
            self.button_frame.pack(fill="x")

            self.prev_btn = tk.Button(self.button_frame, text="<< Previous", command=self.prev_image, width=15)
            self.prev_btn.pack(side=tk.LEFT, padx=20)

            self.no_mining_btn = tk.Button(self.button_frame, text="No Sand Mining (0)", command=lambda: self.set_label(0), width=20, height=2, bg="lightgreen")
            self.no_mining_btn.pack(side=tk.LEFT, padx=10)

            self.mining_btn = tk.Button(self.button_frame, text="Sand Mining (1)", command=lambda: self.set_label(1), width=20, height=2, bg="salmon")
            self.mining_btn.pack(side=tk.LEFT, padx=10)

            self.skip_btn = tk.Button(self.button_frame, text="Skip/Unlabel (?)", command=lambda: self.set_label(-1), width=15)
            self.skip_btn.pack(side=tk.LEFT, padx=10)

            self.next_btn = tk.Button(self.button_frame, text="Next >>", command=self.next_image, width=15)
            self.next_btn.pack(side=tk.RIGHT, padx=20)

            self.status_label = tk.Label(self.root, text="", font=("Arial", 12), fg="blue", anchor='w') # Anchor West
            self.status_label.pack(side=tk.BOTTOM, fill="x", padx=10, pady=5) # Add padding


            # Bind keys
            self.root.bind("<Left>", lambda e: self.prev_image())
            self.root.bind("<Right>", lambda e: self.next_image())
            self.root.bind("0", lambda e: self.set_label(0))
            self.root.bind("1", lambda e: self.set_label(1))
            self.root.bind("?", lambda e: self.set_label(-1)) # Or use <KeyPress-question>
            self.root.bind("<Escape>", lambda e: self.save_and_exit())


            # Delay initial image load slightly to allow canvas to get size
            self.root.after(100, self.load_image)
            self.root.protocol("WM_DELETE_WINDOW", self.save_and_exit) # Save on close
            self.root.mainloop()

        def _load_labels(self):
            labels = {}
            if os.path.exists(self.labels_file):
                try:
                    df = pd.read_csv(self.labels_file)
                    # Handle potential missing columns or different naming
                    if 'filename' in df.columns and 'label' in df.columns:
                         # Ensure filename is string, handle potential float conversion if read incorrectly
                         df['filename'] = df['filename'].astype(str)
                         labels = pd.Series(df.label.values, index=df.filename).to_dict()
                         # Convert labels explicitly to int, handling potential NaNs or other types
                         for img_file, label in labels.items():
                             try:
                                 # Use pd.isna for robust NaN checking
                                 labels[img_file] = int(label) if pd.notna(label) else -1 # Use -1 for unlabeled/NaN
                             except (ValueError, TypeError):
                                 labels[img_file] = -1 # Default to unlabeled if conversion fails
                    else:
                        print(f"Warning: Labels file {self.labels_file} missing 'filename' or 'label' column.")
                except pd.errors.EmptyDataError:
                     print(f"Warning: Labels file {self.labels_file} is empty.") # Handle empty file case
                except Exception as e:
                    print(f"Warning: Could not load labels from {self.labels_file}: {e}")
                    # If loading fails, ask to overwrite or backup
                    if messagebox.askyesno("Label File Error", f"Error loading labels file:\n{e}\n\nDo you want to start with fresh labels? (This will overwrite {self.labels_file} on save)"):
                        labels = {}
                    else:
                        # Optional: Backup existing file
                        try:
                            backup_name = self.labels_file + f".bak_{datetime.now():%Y%m%d_%H%M%S}"
                            shutil.copy2(self.labels_file, backup_name)
                            print(f"Backed up existing labels file to {backup_name}")
                            labels = {} # Start fresh after backup
                        except Exception as backup_err:
                            print(f"Error backing up labels file: {backup_err}")
                            messagebox.showerror("Backup Error", "Could not back up the labels file. Please check permissions or disk space.")
                            labels = {} # Force fresh start if backup fails


            # Initialize missing labels for images currently in the folder
            for img_file in self.image_files:
                if img_file not in labels:
                    labels[img_file] = -1 # -1 indicates unlabeled
            return labels

        def on_canvas_resize(self, event):
             # Rescale and redraw the image when the canvas size changes
             # Add a small delay to prevent excessive redraws during rapid resizing
             if hasattr(self, '_resize_job'):
                 self.root.after_cancel(self._resize_job)
             self._resize_job = self.root.after(100, self.display_image)


        def load_image(self):
             """Loads the image file but doesn't display it yet."""
             if not self.image_files: return
             self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
             img_file = self.image_files[self.current_index]
             img_path = os.path.join(self.images_folder, img_file)

             try:
                 # Only load the image object here
                 self.current_image_object = Image.open(img_path)
                 self.display_image() # Call display function
                 self.update_status()

             except FileNotFoundError:
                 messagebox.showerror("Error Loading Image", f"Image file not found: {img_path}\nIt might have been moved or deleted.")
                 self.canvas.delete("all")
                 self.current_image_object = None
                 self.status_label.config(text=f"Error loading image: {img_file}", fg="red")
             except Exception as e:
                 messagebox.showerror("Error Loading Image", f"Could not load image file: {img_path}\n{type(e).__name__}: {e}")
                 self.canvas.delete("all") # Clear canvas on error
                 self.current_image_object = None
                 self.status_label.config(text=f"Error loading image: {img_file}", fg="red")


        def display_image(self):
             """Resizes and displays the currently loaded image object on the canvas."""
             if self.current_image_object is None:
                 if self._image_id_on_canvas: self.canvas.delete(self._image_id_on_canvas) # Clear previous image if load failed
                 return

             try:
                 img = self.current_image_object # Use the loaded image

                 # Resize image to fit canvas while maintaining aspect ratio
                 canvas_width = self.canvas.winfo_width()
                 canvas_height = self.canvas.winfo_height()

                 if canvas_width < 2 or canvas_height < 2 : # Canvas might not be ready or too small
                      return

                 img_ratio = img.width / img.height
                 canvas_ratio = canvas_width / canvas_height

                 if img_ratio > canvas_ratio: # Image wider than canvas
                     new_width = canvas_width
                     new_height = int(new_width / img_ratio)
                 else: # Image taller than canvas
                     new_height = canvas_height
                     new_width = int(new_height * img_ratio)

                 # Ensure dimensions are at least 1, prevent zero dimension errors
                 new_width = max(1, new_width - 10) # Add padding
                 new_height = max(1, new_height - 10) # Add padding

                 # Use ANTIALIAS for potentially better quality on downscaling than LANCZOS
                 img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                 # Keep reference to PhotoImage to prevent garbage collection
                 self._photo_ref = ImageTk.PhotoImage(img_resized)

                 # Delete *only* the previous image item, not "all"
                 if self._image_id_on_canvas: self.canvas.delete(self._image_id_on_canvas)
                 # Center the image
                 self._image_id_on_canvas = self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self._photo_ref)

             except Exception as e:
                  # Handle potential errors during resizing/display
                  print(f"Error displaying image: {e}")
                  if self._image_id_on_canvas: self.canvas.delete(self._image_id_on_canvas)
                  self.status_label.config(text=f"Error displaying image", fg="red")


        def update_status(self):
            if not self.image_files: return
            img_file = self.image_files[self.current_index]
            label = self.labels.get(img_file, -1)
            label_text = {0: "No Sand Mining", 1: "Sand Mining", -1: "Unlabeled"}[label]
            label_color = {0: "dark green", 1: "red", -1: "dark grey"}[label] # Use darker colors for text

            total_images = len(self.image_files)
            labeled_count = sum(1 for lbl in self.labels.values() if lbl != -1)

            # Truncate long filenames in label
            display_filename = img_file if len(img_file) < 50 else img_file[:25] + "..." + img_file[-20:]
            self.progress_label.config(text=f"Image {self.current_index + 1} of {total_images} ({labeled_count} Labeled) - File: {display_filename}")
            self.status_label.config(text=f"Current Label: {label_text}", fg=label_color)

             # Update button states (optional: visually indicate current label)
            self.no_mining_btn.config(relief=tk.RAISED)
            self.mining_btn.config(relief=tk.RAISED)
            self.skip_btn.config(relief=tk.RAISED)
            if label == 0: self.no_mining_btn.config(relief=tk.SUNKEN)
            elif label == 1: self.mining_btn.config(relief=tk.SUNKEN)
            elif label == -1: self.skip_btn.config(relief=tk.SUNKEN)


        def set_label(self, label):
            if not self.image_files: return
            img_file = self.image_files[self.current_index]
            self.labels[img_file] = label
            self.update_status()
            # Automatically move to next image after labeling
            self.next_image() # Go to next immediately

        def next_image(self):
            if not self.image_files: return
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                self.load_image() # Load the new image

        def prev_image(self):
            if not self.image_files: return
            if self.current_index > 0:
                self.current_index -= 1
                self.load_image() # Load the new image

        def save_labels(self):
            # Only save if there are labels to save
            if not self.labels:
                print("No labels to save.")
                return True # Consider it successful if there's nothing to do

            try:
                # Filter labels to only include files currently in the image list
                current_files_set = set(self.image_files)
                filtered_labels = {k: v for k, v in self.labels.items() if k in current_files_set}

                if not filtered_labels:
                     print("No labels correspond to current image files.")
                     # Optionally clear the labels file if desired, or just don't write anything
                     # with open(self.labels_file, 'w') as f: f.write("filename,label\n") # Clear file
                     return True


                df = pd.DataFrame(filtered_labels.items(), columns=['filename', 'label'])
                # Filter out rows where filename might be NaN or empty (shouldn't happen with filter)
                df = df.dropna(subset=['filename'])
                df = df[df['filename'] != '']
                # Convert label column to integer type before saving, handle potential errors
                df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)

                df.to_csv(self.labels_file, index=False)
                print(f"Labels saved to {self.labels_file}")
                return True
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save labels to {self.labels_file}\n{e}")
                return False

        def save_and_exit(self):
             if self.root is None: # Check if GUI initialization failed
                 return
             if self.save_labels():
                  self.root.destroy()
             else:
                  # Ask user if they want to exit anyway without saving
                  if messagebox.askyesno("Exit Confirmation", "Failed to save labels. Do you still want to exit? (Changes will be lost)"):
                       self.root.destroy()


    def label_images(self):
        """Starts the image labeling GUI."""
        # Check if images were actually downloaded
        try:
             image_files_exist = any(f.lower().endswith('.png') for f in os.listdir(self.images_folder))
        except FileNotFoundError:
             image_files_exist = False # Folder doesn't exist yet

        if not image_files_exist:
             messagebox.showerror("Labeling Error", f"No training images (.png) found in:\n{self.images_folder}\n\nPlease ensure images were downloaded successfully before labeling.")
             return # Don't start GUI if no images

        print("\nStarting Image Labeling GUI...")
        print("Instructions:")
        print("  - Use buttons or keys (0=No Mining, 1=Mining, ?=Skip) to label.")
        print("  - Use Left/Right arrow keys or buttons to navigate.")
        print("  - Press ESC or close window to save and exit.")
        try:
             gui = self._LabelingGUI(self) # Pass the detector instance
             # If GUI failed to initialize (e.g., no images found), gui.root will be None
             if gui.root is None:
                  print("GUI could not be initialized.")
        except tk.TclError as e:
             print(f"\nError initializing Tkinter GUI: {e}")
             print("This might happen if you are running in an environment without a display (e.g., a remote server via SSH without X forwarding).")
             print("Labeling requires a graphical interface.")


    # --- Feature Extraction ---
    def extract_features(self, image_path):
        """
        Extracts features from a single image file.
        *** THIS IS A BASIC PLACEHOLDER - REPLACE WITH BETTER FEATURES ***
        """
        expected_feature_size = 11 # 3 mean + 3 std + 5 GLCM
        try:
            # Ensure file exists before trying to open
            if not os.path.exists(image_path):
                 print(f"  Error: Image file not found during feature extraction: {image_path}")
                 return np.zeros(expected_feature_size)

            img = Image.open(image_path).convert('RGB')
            img_arr = np.array(img)

            # 1. Basic Color Stats
            mean_rgb = np.mean(img_arr, axis=(0, 1))
            std_rgb = np.std(img_arr, axis=(0, 1))

            # 2. Basic Texture (GLCM) - Example
            # Convert to grayscale for GLCM
            img_gray = np.array(img.convert('L'))
            # Ensure image is not flat gray (std dev > 0) before GLCM
            if img_gray.std() > 1e-5: # Use small threshold instead of 0
                 glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                 contrast = graycoprops(glcm, 'contrast')[0, 0]
                 dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                 homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                 energy = graycoprops(glcm, 'energy')[0, 0]
                 correlation = graycoprops(glcm, 'correlation')[0, 0]
            else:
                 # Handle flat images - return zeros or typical values
                 contrast, dissimilarity, homogeneity, energy, correlation = 0, 0, 1, 1.0/max(1, img_gray.size), 0 # Avoid division by zero


            # Combine features into a single vector
            # Ensure the order here MATCHES the order used during prediction!
            features = np.concatenate([
                mean_rgb,
                std_rgb,
                [contrast, dissimilarity, homogeneity, energy, correlation]
            ])

            # Handle potential NaN/inf values (e.g., if std dev is zero, or correlation undefined)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Final check on feature vector size
            if len(features) != expected_feature_size:
                 print(f"  Error: Feature vector size mismatch for {os.path.basename(image_path)}. Expected {expected_feature_size}, got {len(features)}.")
                 return np.zeros(expected_feature_size)


            return features

        except Exception as e:
            print(f"  Error extracting features from {os.path.basename(image_path)}: {type(e).__name__} - {e}")
            return np.zeros(expected_feature_size) # Return zeros if any error


    # --- Model Training ---
    def train_model(self):
        """Trains a classifier based on labeled images and extracted features."""
        print("\nStarting Model Training...")
        if not os.path.exists(self.labels_file):
            print(f"Error: Labels file not found: {self.labels_file}")
            print("Please run the labeling step first.")
            return None

        try:
            labels_df = pd.read_csv(self.labels_file)
            # Filter out unlabeled images (label == -1) and potential NaN filenames
            labels_df = labels_df.dropna(subset=['filename'])
            labels_df['filename'] = labels_df['filename'].astype(str) # Ensure filename is string
            labeled_df = labels_df[(labels_df['label'] != -1) & (labels_df['filename'] != '')].copy()

            # Convert label column safely
            labeled_df['label'] = pd.to_numeric(labeled_df['label'], errors='coerce').fillna(-1).astype(int)
            # Re-filter after conversion, just in case
            labeled_df = labeled_df[labeled_df['label'] != -1]


            if len(labeled_df) < 10: # Need a minimum number of labeled samples
                print(f"Error: Insufficient labeled data ({len(labeled_df)} valid samples found in {self.labels_file}). Need at least 10 with labels 0 or 1.")
                print("Please label more images using the GUI.")
                return None

            # Check class balance
            class_counts = labeled_df['label'].value_counts()
            print("Class distribution in training data:")
            print(class_counts)
            if len(class_counts) < 2:
                 print("Error: Training data contains only one class (0 or 1). Cannot train model.")
                 print("Please ensure you have labeled examples for both 'Sand Mining' and 'No Sand Mining'.")
                 return None
            minority_class_count = class_counts.min()
            # Lower warning threshold slightly if sample size is very small overall
            min_samples_warn = 3 if len(labeled_df) < 20 else 5
            if minority_class_count < min_samples_warn:
                 print(f"Warning: Minority class has only {minority_class_count} samples. Model performance may be poor or unstable.")


            print(f"Extracting features for {len(labeled_df)} labeled images...")
            features_list = []
            valid_labels = []
            filenames = []
            skipped_count = 0

            for idx, row in tqdm(labeled_df.iterrows(), total=len(labeled_df), desc="Extracting Features"):
                img_path = os.path.join(self.images_folder, row['filename'])
                # No need to check os.path.exists here, extract_features handles it
                features = self.extract_features(img_path)

                # Check if feature extraction was successful (returned non-zero vector)
                if features is not None and features.any(): # Check if any feature is non-zero
                   features_list.append(features)
                   valid_labels.append(int(row['label']))
                   filenames.append(row['filename'])
                else:
                    # tqdm.write(f"  Skipping image {row['filename']} due to feature extraction error or all-zero features.")
                    skipped_count += 1


            if skipped_count > 0:
                 print(f"Note: Skipped {skipped_count} images during feature extraction (file not found or error).")

            if not features_list:
                 print("Error: Failed to extract valid features from any labeled image.")
                 return None

            X = np.array(features_list)
            y = np.array(valid_labels)

            # Re-check class balance *after* feature extraction
            final_class_counts = pd.Series(y).value_counts()
            if len(final_class_counts) < 2:
                 print("Error: Training data contains only one class after feature extraction. Cannot train model.")
                 return None
            final_minority_count = final_class_counts.min()
            min_samples_split = 2 # sklearn needs at least 2 samples per class for stratification
            if final_minority_count < min_samples_split:
                 print(f"Warning: Minority class has only {final_minority_count} samples after feature extraction. Stratified split will fail.")


            print(f"Training model with {X.shape[0]} samples and {X.shape[1]} features.")

            # Data Scaling
            print("Scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Save the scaler
            joblib.dump(scaler, self.scaler_file)
            print(f"Feature scaler saved to: {self.scaler_file}")


            # Split data
            run_evaluation = False
            X_train, X_test, y_train, y_test = None, None, None, None
            # Check if enough samples *per class* exist for split
            if X.shape[0] >= 4 and final_minority_count >= min_samples_split:
                try:
                    # Ensure test_size results in at least 1 sample per class in test if possible
                    # A simple approach: fixed test size like 0.25
                    test_size = 0.25
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=y
                    )
                    # Verify test set has both classes if stratification succeeded
                    if len(np.unique(y_test)) == 2:
                        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
                        run_evaluation = True
                    else:
                         print("Warning: Stratified split resulted in only one class in the test set. Training on all data.")
                         run_evaluation = False
                except ValueError as e:
                     print(f"Warning: Could not stratify data for train/test split (likely too few samples in one class): {e}")
                     run_evaluation = False

            if not run_evaluation:
                 print("Training final model on all data without evaluation split.")
                 X_train, y_train = X_scaled, y # Use all data for training

            # Train RandomForest Classifier
            print("Training RandomForestClassifier...")
            # Add class_weight='balanced' for imbalanced datasets
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1, min_samples_leaf=2) # Add min_samples_leaf
            model.fit(X_train, y_train)

            # Evaluate model if split was possible and successful
            if run_evaluation and X_test is not None and y_test is not None:
                 print("Evaluating model on test set...")
                 y_pred = model.predict(X_test)
                 accuracy = accuracy_score(y_test, y_pred)
                 # Specify zero_division=0 for report
                 report = classification_report(y_test, y_pred, zero_division=0)

                 print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
                 print("Classification Report:")
                 print(report)


            # Save the model trained (either on train split or all data if split failed)
            final_model = model
            joblib.dump(final_model, self.model_file)
            print(f"\nTrained model saved successfully to: {self.model_file}")

            return final_model

        except FileNotFoundError:
            print(f"Error: Could not find label file at {self.labels_file}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: Label file {self.labels_file} is empty or unreadable.")
            return None
        except KeyError as e:
            print(f"Error: Missing expected column in label file {self.labels_file}: {e}")
            return None
        except Exception as e:
            print(f"An error occurred during model training: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return None


# --- Sand Mining Probability Mapper Class ---
class SandMiningProbabilityMapper:
    def __init__(self, model_path=None, scaler_path=None, feature_extractor_func=None):
        # Ensure EE is initialized for the mapper instance too
        self.ee_initialized = initialize_ee()
        if not self.ee_initialized:
            raise Exception("Earth Engine could not be initialized for Mapper. Exiting.")

        self.base_folder = BASE_FOLDER
        self.output_folder = PROBABILITY_MAPS_FOLDER
        self.temp_folder = TEMP_FOLDER
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        print(f"Probability maps will be saved in: {self.output_folder}")
        print(f"Temporary processing files in: {self.temp_folder}")

        # --- Model and Scaler Loading ---
        if model_path is None: model_path = MODEL_FILE
        if scaler_path is None: scaler_path = SCALER_FILE

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error: Model file not found at {model_path}. Please train the model first.")
        if not os.path.exists(scaler_path):
             raise FileNotFoundError(f"Error: Feature scaler file not found at {scaler_path}. It should be created during training.")

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Feature scaler loaded successfully from {scaler_path}")
        except Exception as e:
            raise Exception(f"Error loading model or scaler: {e}")

        # --- Feature Extractor ---
        # Use the provided function or default to the one from SandMiningDetection
        if feature_extractor_func:
             self.extract_features = feature_extractor_func
        else:
             # Create a temporary instance to get the method
             # This isn't ideal, maybe refactor extract_features to be standalone
             temp_detector = SandMiningDetection()
             self.extract_features = temp_detector.extract_features
             print("Using default feature extractor from SandMiningDetection class.")


    def get_latest_imagery_date(self, region):
        """Determine the latest available imagery date with acceptable cloud coverage"""
        if not self.ee_initialized: return datetime.now().strftime('%Y-%m-%d'), "Error - EE Not Init"
        try:
            # Use Python's datetime for current date calculation
            today_dt = datetime.now()
            today_ee = ee.Date(today_dt) # Convert to EE Date object
            latest_date_info = {'date': None, 'source': 'None', 'cloud': 999}

            # Check Sentinel-2 Harmonized (last 6 months)
            start_s2_dt = today_dt - timedelta(days=180)
            start_s2_ee = ee.Date(start_s2_dt)

            s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region) \
                .filterDate(start_s2_ee, today_ee) \
                .sort('system:time_start', False) # Sort descending by date

            # Iterate through images to find the best one (least cloudy recent)
            image_list_s2 = s2_col.toList(20) # Check the latest 20 images first
            for i in range(image_list_s2.size().getInfo()):
                 img = ee.Image(image_list_s2.get(i))
                 cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE')
                 # Need to getInfo() to check the value client-side
                 if cloud_cover is not None:
                      # Add error handling for getInfo() calls
                      try:
                          cloud_cover_val = cloud_cover.getInfo()
                          if cloud_cover_val is not None and cloud_cover_val < 35:
                               date_millis = img.get('system:time_start').getInfo()
                               latest_date_info = {
                                    'date': datetime.fromtimestamp(date_millis/1000),
                                    'source': 'Sentinel-2',
                                    'cloud': cloud_cover_val
                               }
                               break # Found a good recent one
                      except ee.EEException as getinfo_err:
                          print(f"Warning: Error getting cloud cover info: {getinfo_err}")
                          continue # Skip image if metadata cannot be retrieved


            # If no recent good S2, check Landsat 9 (last 12 months)
            if latest_date_info['date'] is None:
                start_l9_dt = today_dt - timedelta(days=365)
                start_l9_ee = ee.Date(start_l9_dt)

                l9_col = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
                    .filterBounds(region) \
                    .filterDate(start_l9_ee, today_ee) \
                    .sort('system:time_start', False)

                image_list_l9 = l9_col.toList(10)
                for i in range(image_list_l9.size().getInfo()):
                    img = ee.Image(image_list_l9.get(i)) # Corrected: use l9 list
                    cloud_cover = img.get('CLOUD_COVER') # Different metadata name
                    if cloud_cover is not None:
                         try:
                              cloud_cover_val = cloud_cover.getInfo()
                              if cloud_cover_val is not None and cloud_cover_val < 40:
                                   date_millis = img.get('system:time_start').getInfo()
                                   ld_date = datetime.fromtimestamp(date_millis/1000)
                                   latest_date_info = {
                                        'date': ld_date,
                                        'source': 'Landsat 9',
                                        'cloud': cloud_cover_val
                                   }
                                   break # Found a suitable Landsat image
                         except ee.EEException as getinfo_err:
                              print(f"Warning: Error getting cloud cover info for Landsat: {getinfo_err}")
                              continue

            if latest_date_info['date']:
                return latest_date_info['date'].strftime('%Y-%m-%d'), f"{latest_date_info['source']} ({latest_date_info['cloud']:.1f}% cloud)"
            else:
                # Fallback if absolutely nothing found
                return today_dt.strftime('%Y-%m-%d'), "No recent imagery found" # Use Python date

        except ee.EEException as e:
             print(f"EE Error determining latest imagery date: {e}")
             return datetime.now().strftime('%Y-%m-%d'), "Error - EE Exception"
        except Exception as e:
             print(f"Error determining latest imagery date: {type(e).__name__} - {e}")
             return datetime.now().strftime('%Y-%m-%d'), "Error - General Exception"

    def generate_river_points(self, shapefile_path, distance_km=1.0):
        """Generate dense sampling points along the entire river geometry."""
        print(f"\nLoading shapefile for mapping: {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs is None:
                print("Warning: Shapefile CRS not set. Assuming WGS84 (EPSG:4326).")
                gdf.crs = "EPSG:4326"
            elif gdf.crs.to_epsg() != 4326:
                print(f"Converting CRS from {gdf.crs.to_string()} to WGS84 (EPSG:4326)")
                gdf = gdf.to_crs("EPSG:4326")

            # Simplify and combine geometries
            simplified_geoms = []
            invalid_geom_count = 0
            for geom in gdf.geometry:
                 if geom is None: continue # Skip null geometries
                 if not geom.is_valid:
                     invalid_geom_count +=1
                     # Attempt to fix invalid geometry
                     try:
                          geom_fixed = geom.buffer(0)
                          if geom_fixed.is_valid and not geom_fixed.is_empty:
                               geom = geom_fixed # Use the fixed geometry
                          else:
                               continue # Skip if buffer(0) didn't help or resulted in empty
                     except Exception as buf_err:
                          # print(f"Warning: Could not fix invalid geometry with buffer(0): {buf_err}") # Reduce noise
                          continue # Skip this geometry if buffer fails
                 if geom.is_empty: continue # Skip empty geometries

                 # Simplify slightly to speed up interpolation and reduce complexity
                 simplified = geom.simplify(0.0001, preserve_topology=True)
                 if simplified.is_empty: continue # Skip if simplification results in empty

                 if isinstance(simplified, (Polygon, MultiPolygon)):
                     # Boundary might be complex or empty, handle carefully
                     boundary = simplified.boundary
                     # Check if boundary is LineString or MultiLineString before adding
                     if isinstance(boundary, (LineString, MultiLineString)) and not boundary.is_empty:
                          simplified_geoms.append(boundary)
                 elif isinstance(simplified, (LineString, MultiLineString)):
                     simplified_geoms.append(simplified)

            if invalid_geom_count > 0:
                 print(f"Note: Found and attempted to fix {invalid_geom_count} invalid geometries.")

            if not simplified_geoms:
                 print("Error: No valid LineString or Polygon boundary geometries found after cleaning/simplification.")
                 return [], None # Return GDF as None here too

            # Perform unary_union
            try:
                 full_river_geom = unary_union(simplified_geoms)
            except Exception as union_err:
                 print(f"Error during unary_union: {union_err}. Check input geometries.")
                 return [], gdf # Return original GDF if union fails


            if full_river_geom.is_empty:
                 print("Error: Combined river geometry is empty after unary_union.")
                 return [], gdf # Return original GDF

            coordinates = []
            # Use a small epsilon to avoid division by zero if distance_km is very small
            distance_degrees = distance_km / 111.32 if distance_km > 1e-6 else 1e-6

            print(f"Generating mapping points every {distance_km:.2f} km along the river...")
            print(f"Combined geometry type: {full_river_geom.geom_type}, Length (degrees): {full_river_geom.length:.4f}")


            # *** CORRECTED TYPE CHECKING using shapely.geometry ***
            target_geoms = []
            # Check specific Shapely types first
            if isinstance(full_river_geom, shapely.geometry.LineString):
                 target_geoms = [full_river_geom]
            elif isinstance(full_river_geom, shapely.geometry.MultiLineString):
                 target_geoms = list(full_river_geom.geoms)
            # Check for GeometryCollection LAST, as LineString/MultiLineString are also geometries
            elif isinstance(full_river_geom, shapely.geometry.GeometryCollection):
                 print("Combined geometry is a Collection, processing LineString components.")
                 for geom_part in full_river_geom.geoms: # Iterate through components
                     if isinstance(geom_part, shapely.geometry.LineString):
                         if not geom_part.is_empty: target_geoms.append(geom_part)
                     elif isinstance(geom_part, shapely.geometry.MultiLineString):
                          if not geom_part.is_empty:
                              target_geoms.extend([line for line in geom_part.geoms if not line.is_empty]) # Add individual lines
            # Handle other possibilities like MultiPoint, Point, Polygon (though unlikely after boundary/union)
            else:
                print(f"Warning: Combined geometry is an unhandled type for point generation: {full_river_geom.geom_type}.")


            # Interpolate points along the collected LineString geometries
            total_points_generated = 0
            for line_geom in target_geoms:
                 # Double-check it's a valid LineString before proceeding
                 if isinstance(line_geom, shapely.geometry.LineString) and not line_geom.is_empty:
                     line_len = line_geom.length
                     if line_len > 1e-9: # Check length is meaningfully positive
                         # Use ceiling to ensure endpoint is included
                         num_segments = max(1, int(np.ceil(line_len / distance_degrees)))
                         # Interpolate from 0 to num_segments (inclusive)
                         for i in range(num_segments + 1):
                             fraction = float(i) / num_segments if num_segments > 0 else 0.0
                             # Ensure fraction is within [0, 1]
                             fraction = max(0.0, min(1.0, fraction))
                             try:
                                  point = line_geom.interpolate(fraction, normalized=True)
                                  if isinstance(point, Point): # Ensure result is a Point
                                      # Ensure coordinates are valid numbers
                                      if np.isfinite(point.y) and np.isfinite(point.x):
                                           coordinates.append((point.y, point.x)) # lat, lon
                                           total_points_generated +=1
                                      # else: print(f"Warning: Interpolated NaN coordinates for fraction {fraction}") # Reduce noise
                                  # else: print(f"Warning: Interpolation did not return a Point for fraction {fraction}") # Reduce noise
                             except Exception as interp_err:
                                  print(f"Warning: Error interpolating point on line segment: {interp_err}")

            # Remove duplicate points
            if not coordinates:
                 print("Warning: No coordinates generated after processing geometries.")
                 # Check if target_geoms was empty, indicating issue with geometry types
                 if not target_geoms:
                      print("  This might be due to the combined geometry type not being LineString, MultiLineString, or GeometryCollection containing lines.")
                 return [], gdf # Return original GDF
            unique_coordinates = sorted(list(set(coordinates)))

            print(f"Generated {len(unique_coordinates)} unique mapping points.")

            # Save mapping points (optional)
            # points_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in unique_coordinates], crs=gdf.crs)
            # points_shapefile = os.path.join(self.output_folder, 'river_mapping_points.shp')
            # try:
            #     points_gdf.to_file(points_shapefile)
            #     print(f"Mapping points saved to: {points_shapefile}")
            # except Exception as save_err:
            #     print(f"Warning: Could not save mapping points shapefile: {save_err}")


            return unique_coordinates, gdf # Return points and original GDF

        except ImportError:
             print("Error: Shapely or Geopandas library not found or import failed. Please ensure they are installed.")
             return [], None
        except Exception as e:
            print(f"Error generating river points for mapping: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return [], None # Return GDF as None on error


    def analyze_point(self, lat, lon, latest_date_str, img_dim=DEFAULT_MAP_IMAGE_DIM, buffer_m=DEFAULT_BUFFER_METERS):
        """Download image, extract features, and predict probability for a single point."""
        if not self.ee_initialized: return None
        temp_file = None # Initialize temp_file to None
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(buffer_m)

            # Define date range around the latest identified date
            # Use Python datetime for calculations before creating EE Dates
            try:
                latest_dt = datetime.strptime(latest_date_str, '%Y-%m-%d')
            except ValueError:
                print(f"Warning: Invalid latest_date_str '{latest_date_str}'. Using current date.")
                latest_dt = datetime.now()
                latest_date_str = latest_dt.strftime('%Y-%m-%d') # Update string


            start_dt_narrow = latest_dt - timedelta(days=15)
            end_dt = latest_dt + timedelta(days=1) # Include target day

            latest_date_ee = ee.Date(latest_dt)
            start_date_ee_narrow = ee.Date(start_dt_narrow)
            end_date_ee = ee.Date(end_dt)


            # Get the best image within this narrow window
            s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region) \
                .filterDate(start_date_ee_narrow, end_date_ee) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE') # Least cloudy first

            image_to_use = s2_collection.first() # Get the best available image

            if image_to_use is None:
                 # If nothing found in the narrow window, broaden search slightly
                 start_dt_broad = latest_dt - timedelta(days=45)
                 start_date_ee_broad = ee.Date(start_dt_broad)

                 s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                     .filterBounds(region) \
                     .filterDate(start_date_ee_broad, end_date_ee) \
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
                     .sort('CLOUDY_PIXEL_PERCENTAGE')
                 image_to_use = s2_collection.first()

            # Ensure image_to_use is valid before proceeding
            if image_to_use is None:
                return None # Skip this point if no image available
            image_to_use = ee.Image(image_to_use) # Ensure it's cast to ee.Image


            # Get actual date of the image used
            try:
                 actual_img_date_millis = image_to_use.get('system:time_start').getInfo()
                 actual_img_date = datetime.fromtimestamp(actual_img_date_millis/1000).strftime('%Y-%m-%d')
            except ee.EEException as date_err:
                 tqdm.write(f"Warning: Could not get image date: {date_err}")
                 actual_img_date = "Unknown" # Fallback date


            # Define visualization parameters (True Color)
            vis_params = {
                'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4
            }

            # Get download URL
            # Add error handling for getInfo() on region
            try:
                region_info = region.getInfo()
                if not region_info or 'coordinates' not in region_info:
                     # tqdm.write(f"Warning: Could not get region info for point {lat:.4f}, {lon:.4f}") # Reduce noise
                     return None
                region_coords = region_info['coordinates']
            except ee.EEException as region_err:
                 tqdm.write(f"Warning: Error getting region info ({lat:.4f}, {lon:.4f}): {region_err}")
                 return None


            download_url = image_to_use.getThumbURL({
                **vis_params,
                'region': region_coords,
                'dimensions': img_dim,
                'format': 'png'
            })

            # Download image
            response = requests.get(download_url, timeout=90) # Extended timeout
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert('RGB')

            # Save temporarily for feature extraction
            # Ensure temp folder exists (though created in init)
            os.makedirs(self.temp_folder, exist_ok=True)
            temp_filename = f'map_temp_{lat:.6f}_{lon:.6f}.png'
            temp_file = os.path.join(self.temp_folder, temp_filename)
            img.save(temp_file)

            # Extract features using the *same* method as training
            features = self.extract_features(temp_file)

            if features is None or not features.any(): # Check if feature extraction failed or returned zeros
                 # tqdm.write(f"  Warning: Feature extraction failed for point ({lat:.4f}, {lon:.4f}).") # Reduce noise
                 return None

            # Scale features using the *loaded* scaler
            X_scaled = self.scaler.transform([features]) # Scale a single sample

            # Predict probability (probability of class 1)
            probability = self.model.predict_proba(X_scaled)[0][1]

            return probability, actual_img_date # Return probability and actual date used

        except ee.EEException as ee_err:
             ee_err_str = str(ee_err).lower()
             # Only print if it's not a common, handled error type
             if not any(term in ee_err_str for term in ['quota', 'rate limit', 'timed out', 'backend error', 'memory limit', 'too many requests']):
                 tqdm.write(f"  EE Error analyzing point ({lat:.4f}, {lon:.4f}): {ee_err_str[:150]}...")
             # Backoff for common errors
             if any(term in ee_err_str for term in ['quota', 'rate limit', 'user memory limit', 'computation timed out', 'backend error', 'too many requests']):
                 wait_time = min(64, 2**(np.random.randint(1, 7))) # Random backoff 2-64s
                 time.sleep(wait_time)
             return None
        except requests.exceptions.RequestException as req_err:
             # Don't flood logs with download errors during mapping, maybe sample them
             if np.random.rand() < 0.1: # Log ~10% of download errors
                  tqdm.write(f"  Download Error ({lat:.4f}, {lon:.4f}): {req_err}")
             time.sleep(1) # Shorter wait after download errors
             return None
        except Exception as e:
             tqdm.write(f"  Unexpected error analyzing point ({lat:.4f}, {lon:.4f}): {type(e).__name__} - {e}")
             return None
        finally:
             # Clean up temporary file
             if temp_file and os.path.exists(temp_file):
                 try:
                     os.remove(temp_file)
                 except Exception:
                      pass # Ignore errors removing temp file


    def create_probability_map(self, shapefile_path, distance_km=1.0, output_file=None):
        """Generate the comprehensive probability map."""
        print("\nStarting Sand Mining Probability Mapping...")

        # 1. Generate dense points along the river
        coords, river_gdf = self.generate_river_points(shapefile_path, distance_km)
        # Check if generate_river_points returned valid data
        if river_gdf is None or not coords: # coords check is important too
            print("Error: Failed to generate river points or load river geometry for mapping.")
            if os.path.exists(self.temp_folder):
                try: shutil.rmtree(self.temp_folder)
                except Exception: pass
            return None

        # 2. Determine the overall latest imagery date for the area
        try:
             bounds = river_gdf.total_bounds # [minx, miny, maxx, maxy]
             # Check if bounds are valid
             if not (np.all(np.isfinite(bounds)) and bounds[0] < bounds[2] and bounds[1] < bounds[3]):
                  print(f"Error: Invalid bounds calculated from shapefile: {bounds}. Cannot determine AOI for imagery search.")
                  if os.path.exists(self.temp_folder):
                      try: shutil.rmtree(self.temp_folder)
                      except Exception: pass
                  return None
             aoi_region = ee.Geometry.Rectangle(list(bounds))
        except Exception as bounds_err:
             print(f"Error creating AOI for imagery search from shapefile bounds: {bounds_err}")
             if os.path.exists(self.temp_folder):
                 try: shutil.rmtree(self.temp_folder)
                 except Exception: pass
             return None


        latest_date_str, source_info = self.get_latest_imagery_date(aoi_region)
        if "Error" in source_info:
             print(f"Warning: Could not reliably determine latest imagery date ({source_info}). Using current date: {latest_date_str}")
        else:
             print(f"Using imagery baseline date: {latest_date_str} (Source: {source_info})")

        # 3. Analyze each point
        results = []
        print(f"\nAnalyzing {len(coords)} points along the river...")
        # Use tqdm for progress bar
        with tqdm(total=len(coords), desc="Mapping Points", unit="point", smoothing=0.1) as pbar:
            for i, (lat, lon) in enumerate(coords):
                analysis_result = self.analyze_point(lat, lon, latest_date_str)

                if analysis_result is not None:
                    probability, actual_img_date = analysis_result
                    results.append({
                        'latitude': lat,
                        'longitude': lon,
                        'probability': probability,
                        'image_date': actual_img_date, # Store actual date used
                        'classification': 'Sand Mining Likely' if probability >= 0.7 else ('Possible Sand Mining' if probability >= 0.5 else 'No Sand Mining Likely')
                    })
                # Add a small sleep *regardless* of success to avoid hitting EE limits too hard
                time.sleep(0.05) # 50ms delay
                pbar.update(1) # Update progress bar

        # 4. Process results and save
        if not results:
            print("\nError: No valid results generated from point analysis. Possible reasons: No suitable imagery found, errors during analysis, or high EE request rate.")
            if os.path.exists(self.temp_folder):
                try: shutil.rmtree(self.temp_folder)
                except Exception: pass
            return None

        results_df = pd.DataFrame(results)
        # Use the baseline date for the main filename, handle potential errors in date string
        try:
             map_filename_date = datetime.strptime(latest_date_str, '%Y-%m-%d').strftime('%Y%m%d')
        except ValueError:
             map_filename_date = datetime.now().strftime('%Y%m%d') # Fallback

        csv_path = os.path.join(self.output_folder, f'sand_mining_probabilities_{map_filename_date}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved probability results to: {csv_path}")

        # 5. Create interactive map
        if output_file is None:
            # Create default filename using shapefile name and date
            shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
            output_file = os.path.join(self.output_folder, f'sand_mining_map_{shapefile_name}_{map_filename_date}.html')

        # Find the actual date range of imagery used
        try:
             min_img_date = results_df['image_date'].min()
             max_img_date = results_df['image_date'].max()
             date_range_str = f"{min_img_date} to {max_img_date}" if min_img_date != max_img_date else min_img_date
             print(f"Actual imagery dates used in map range from: {date_range_str}")
        except Exception as date_err:
             print(f"Warning: Could not determine date range from results: {date_err}")
             date_range_str = "Unknown"

        # Ensure river_gdf is passed correctly
        self.generate_interactive_map(results_df, river_gdf, output_file, date_range_str)


        # Clean up temp folder
        try:
            if os.path.exists(self.temp_folder): # Check again before removing
                 shutil.rmtree(self.temp_folder)
                 print(f"Cleaned up temporary folder: {self.temp_folder}")
        except Exception as e:
            print(f"Warning: Could not clean up temp folder {self.temp_folder}: {e}")


        return results_df

    def generate_interactive_map(self, results_df, river_gdf, output_file, imagery_date_range):
        """Create an interactive Folium map with the results."""
        print("Generating interactive map...")
        if results_df.empty:
             print("Warning: No results to map.")
             return

        # Calculate map center and bounds
        try:
            # Drop rows with NaN lat/lon before calculating bounds/center
            results_df_valid = results_df.dropna(subset=['latitude', 'longitude'])
            if results_df_valid.empty:
                 print("Warning: No valid coordinates in results to determine map center/bounds.")
                 center_lat, center_lon = 0, 0 # Default fallback
                 map_bounds = None
                 zoom_start = 2
            else:
                 center_lat = results_df_valid['latitude'].mean()
                 center_lon = results_df_valid['longitude'].mean()
                 min_lat = results_df_valid['latitude'].min()
                 max_lat = results_df_valid['latitude'].max()
                 min_lon = results_df_valid['longitude'].min()
                 max_lon = results_df_valid['longitude'].max()
                 map_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
                 zoom_start = 11 # Default zoom if bounds exist
        except Exception as center_err:
             print(f"Warning: Error calculating map center/bounds: {center_err}. Using default view.")
             center_lat, center_lon = 0, 0
             map_bounds = None
             zoom_start = 2


        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=None) # Start with no tiles

        # Add base layers
        folium.TileLayer('CartoDB positron', name='CartoDB Positron (Light)', show=True).add_to(m)
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=False).add_to(m)
        folium.TileLayer(
             tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
             attr='Esri',
             name='Esri World Imagery',
             overlay=False,
             control=True,
             show=False # Start with light map active
        ).add_to(m)


        # River Geometry Layer
        if river_gdf is not None and not river_gdf.empty:
            try:
                # Attempt to get column names, handle if it's empty or invalid GDF
                tooltip_cols = list(river_gdf.columns) if not river_gdf.empty else None
                # Exclude geometry column itself from tooltip
                if tooltip_cols and 'geometry' in tooltip_cols: tooltip_cols.remove('geometry')
                # Only include a few relevant columns if too many exist?
                max_tooltip_cols = 5
                if tooltip_cols and len(tooltip_cols) > max_tooltip_cols:
                     tooltip_cols = tooltip_cols[:max_tooltip_cols] # Take first few
                if not tooltip_cols: tooltip_cols = None # Handle case where only geometry exists


                folium.GeoJson(
                    river_gdf,
                    name='River Outline',
                    style_function=lambda x: {'color': 'blue', 'weight': 2, 'opacity': 0.6},
                    tooltip=folium.GeoJsonTooltip(fields=tooltip_cols, aliases=tooltip_cols) if tooltip_cols else None,
                    show=True # Show river by default
                ).add_to(m)
            except Exception as geojson_err:
                print(f"Warning: Could not add river GeoJSON layer: {geojson_err}")
        else:
             print("Warning: River GeoDataFrame is None or empty, skipping GeoJSON layer.")


        # --- Probability Layers ---
        # Colormap
        color_scale = cm.LinearColormap(['#00FF00', '#FFFF00', '#FFA500', '#FF0000'], # Green, Yellow, Orange, Red
                                          vmin=0, vmax=1, caption='Sand Mining Probability')
        m.add_child(color_scale)

        # Heatmap Layer
        heatmap_data = results_df[['latitude', 'longitude', 'probability']].dropna().values.tolist()
        if heatmap_data: # Only add if data exists
            HeatMap(
                heatmap_data,
                name='Probability Heatmap',
                radius=12, # Adjust radius
                blur=8,   # Adjust blur
                gradient={0.0: '#00FF00', 0.5: '#FFFF00', 0.7: '#FFA500', 1: '#FF0000'},
                show=False # Initially hidden
            ).add_to(m)


        # Marker Clusters for different risk levels
        # Only create cluster if data exists for that level
        high_risk_df = results_df[results_df['probability'] >= 0.7].dropna(subset=['latitude', 'longitude'])
        medium_risk_df = results_df[(results_df['probability'] >= 0.5) & (results_df['probability'] < 0.7)].dropna(subset=['latitude', 'longitude'])
        low_risk_df = results_df[results_df['probability'] < 0.5].dropna(subset=['latitude', 'longitude'])

        # --- Add Markers ---
        # Create cluster groups even if empty initially, add markers if data exists
        mc_high = MarkerCluster(name=f"High Risk Points (p >= 0.7) [{len(high_risk_df)}]", show=True).add_to(m)
        mc_medium = MarkerCluster(name=f"Medium Risk Points (0.5 <= p < 0.7) [{len(medium_risk_df)}]", show=True).add_to(m)
        mc_low = MarkerCluster(name=f"Low Risk Points (p < 0.5) [{len(low_risk_df)}]", show=False).add_to(m) # Initially hidden

        if not high_risk_df.empty:
             for idx, row in high_risk_df.iterrows():
                  self._add_marker(mc_high, row, color_scale)

        if not medium_risk_df.empty:
             for idx, row in medium_risk_df.iterrows():
                  self._add_marker(mc_medium, row, color_scale)

        if not low_risk_df.empty:
             for idx, row in low_risk_df.iterrows():
                  self._add_marker(mc_low, row, color_scale)


        # Fit map bounds if they were calculated
        if map_bounds:
             m.fit_bounds(map_bounds, padding=(10, 10)) # Add a little padding

        # Add Layer Control
        folium.LayerControl(collapsed=False).add_to(m)


        # Add Title and Info Box
        num_total = len(results_df)
        num_high = len(high_risk_df)
        num_medium = len(medium_risk_df)
        avg_prob = results_df['probability'].mean() if num_total > 0 else 0
        high_perc = (num_high / num_total * 100) if num_total > 0 else 0
        med_perc = (num_medium / num_total * 100) if num_total > 0 else 0


        title_html = f'''
             <div style="position: fixed;
                        top: 10px; left: 50px; width: 300px; height: auto;
                        background-color: rgba(255, 255, 255, 0.85); /* Semi-transparent white */
                        border:2px solid grey; z-index:9999; font-size:14px;
                        padding: 10px; border-radius: 5px; box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
                 <h4 style="margin-top:0; text-align:center; margin-bottom: 8px;">Sand Mining Probability Map</h4>
                 <p style="font-size:11px; margin-bottom:5px; border-bottom: 1px solid #ccc; padding-bottom: 4px;">
                     <b>Imagery Dates:</b> {imagery_date_range}
                 </p>
                 <ul style="font-size:11px; list-style-type: none; padding-left: 0; margin-bottom: 5px;">
                     <li>Total Points Analyzed: {num_total}</li>
                     <li>Avg. Probability: {avg_prob:.3f}</li>
                     <li style="color:red;">High Risk (p≥0.7): {num_high} ({high_perc:.1f}%)</li>
                     <li style="color:orange;">Medium Risk (0.5≤p<0.7): {num_medium} ({med_perc:.1f}%)</li>
                 </ul>
                 <p style="font-size:10px; text-align:center; margin-bottom:0; color:#555;"><i>Toggle layers via control (top-right)</i></p>
             </div>
             '''
        m.get_root().html.add_child(folium.Element(title_html))


        # Save the map
        try:
             m.save(output_file)
             print(f"Successfully generated map: {output_file}")
        except Exception as save_map_err:
             print(f"Error saving map to {output_file}: {save_map_err}")


    def _add_marker(self, marker_cluster, data_row, color_scale):
         """Helper function to add a CircleMarker to a MarkerCluster."""
         prob = data_row['probability']
         color = color_scale(prob)
         # Ensure lat/lon are valid numbers before creating marker
         lat = data_row['latitude']
         lon = data_row['longitude']
         if not (np.isfinite(lat) and np.isfinite(lon)):
              # print(f"Skipping marker for invalid coordinates: {lat}, {lon}") # Reduce noise
              return

         popup_html = f"""
             <b>Lat:</b> {lat:.5f}, <b>Lon:</b> {lon:.5f}<br>
             <b>Probability:</b> {prob:.3f}<br>
             <b>Classification:</b> {data_row['classification']}<br>
             <b>Image Date:</b> {data_row['image_date']}
         """
         popup = folium.Popup(popup_html, max_width=250)

         marker = folium.CircleMarker(
             location=[lat, lon],
             radius=6, # Smaller radius for individual points
             color='black', # Outline color
             weight=0.5,
             fill=True,
             fill_color=color,
             fill_opacity=0.8,
             popup=popup,
             tooltip=f"P: {prob:.2f}" # Simple tooltip on hover
         )
         marker.add_to(marker_cluster) # Use add_to method


    def create_report(self, results_df, shapefile_path, imagery_date_range, output_file=None):
        """Generate a simple summary plot report."""
        print("Generating summary report plot...")
        if results_df is None or results_df.empty:
            print("Warning: No results to create report from.")
            return None

        if output_file is None:
             # Try to extract start date, handle potential errors
             try:
                 # Use first part of date range for filename
                 map_filename_date = imagery_date_range.split(' ')[0].replace('-', '')
             except:
                 map_filename_date = datetime.now().strftime('%Y%m%d') # Fallback to current date

             # Add shapefile name to report filename
             shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
             output_file = os.path.join(self.output_folder, f'sand_mining_summary_{shapefile_name}_{map_filename_date}.png')


        try:
            plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
            fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # 1 row, 2 columns

            # Histogram of Probabilities
            valid_probs = results_df['probability'].dropna() # Drop NaNs before plotting
            if not valid_probs.empty:
                 axs[0].hist(valid_probs, bins=20, color='skyblue', edgecolor='black')
                 axs[0].axvline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Threshold 0.5')
                 axs[0].axvline(0.7, color='red', linestyle='--', linewidth=1.5, label='Threshold 0.7')
                 axs[0].legend()
            else:
                 axs[0].text(0.5, 0.5, "No valid probability data", ha='center', va='center')

            axs[0].set_title('Distribution of Sand Mining Probabilities')
            axs[0].set_xlabel('Probability')
            axs[0].set_ylabel('Number of Points')
            axs[0].grid(True, linestyle='--', alpha=0.6)


            # Bar Chart of Classifications
            classification_counts = results_df['classification'].value_counts()
            # Define expected classification order for consistent coloring
            class_order = ['No Sand Mining Likely', 'Possible Sand Mining', 'Sand Mining Likely']
             # Define colors using a dictionary for direct mapping
            color_map = {
                 'No Sand Mining Likely': '#90EE90', # lightgreen
                 'Possible Sand Mining': '#FFA500', # orange
                 'Sand Mining Likely': '#FF6347'  # tomato red
             }
            # Reindex counts based on expected order, fill missing with 0
            classification_counts = classification_counts.reindex(class_order, fill_value=0)
            # Get colors in the correct order, default to gray if classification is unexpected
            bar_colors = [color_map.get(cls, 'gray') for cls in classification_counts.index]

            if not classification_counts.empty:
                 bars = classification_counts.plot(kind='bar', ax=axs[1], color=bar_colors, edgecolor='black')
                 axs[1].tick_params(axis='x', rotation=15) # Rotate labels slightly

                 # Add counts on top of bars using the 'bars' object from plot
                 if hasattr(bars, 'patches'): # Check if it's a bar plot with patches
                      for bar in bars.patches:
                          if bar.get_height() > 0: # Only label non-zero bars
                              axs[1].text(bar.get_x() + bar.get_width() / 2,
                                          bar.get_height() + (axs[1].get_ylim()[1] * 0.01), # Adjust position
                                          f'{int(bar.get_height())}', # Display integer count
                                          ha='center', va='bottom', fontsize=9)
            else:
                 axs[1].text(0.5, 0.5, "No classification data", ha='center', va='center')


            axs[1].set_title('Point Classifications')
            axs[1].set_xlabel('Classification')
            axs[1].set_ylabel('Number of Points')
            axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)


            # Overall Figure Title
            try: # Safely get river name
                 river_name = os.path.splitext(os.path.basename(shapefile_path))[0]
            except:
                 river_name = "Unknown River"

            fig.suptitle(f'Sand Mining Analysis Summary - {river_name}\nImagery Date Range: {imagery_date_range}', fontsize=16)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close(fig) # Close the plot figure to free memory

            print(f"Summary plot saved to: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating summary plot: {e}")
            import traceback
            traceback.print_exc()
            return None


# --- Command-Line Interface Logic ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Sand Mining Detection and Mapping Tool v1.4',
        formatter_class=argparse.RawTextHelpFormatter # Keep newlines in help text
    )
    parser.add_argument(
        '--mode', type=str, required=True, choices=['train', 'map', 'both'],
        help="Operating mode:\n"
             "  train - Download images, label them, and train a model.\n"
             "  map   - Create a probability map using an existing model.\n"
             "  both  - Run training first, then create a map."
    )
    parser.add_argument(
        '--shapefile', type=str, required=True,
        help='Path to the river shapefile (.shp).'
    )
    parser.add_argument(
        '--distance', type=float, default=1.0,
        help='Sampling distance in km for mapping points (default: 1.0km).\n'
             'For training, this sets minimum separation between random points.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output filename for the interactive map (e.g., my_river_map.html).\n'
             'If not specified, a default name with date is used.'
    )
    parser.add_argument(
        '--model', type=str, default=MODEL_FILE,
        help=f'Path to the trained model file (.joblib) for mode=map.\n'
             f'(Default: {MODEL_FILE})'
    )
    parser.add_argument(
        '--scaler', type=str, default=SCALER_FILE,
        help=f'Path to the feature scaler file (.joblib) for mode=map.\n'
              f'(Default: {SCALER_FILE})'
    )
    parser.add_argument(
        '--sample-size', type=int, default=30,
        help='Number of random sample points to generate for training (default: 30).'
    )

    # Simple confirmation prompt bypass
    parser.add_argument('-y', '--yes', action='store_true', help='Automatically answer yes to confirmations (e.g., overwriting labels).')


    if len(sys.argv) == 1:
         parser.print_help(sys.stderr)
         sys.exit(1)

    return parser.parse_args()

def run_training(args):
    """Run the training workflow."""
    print("\n" + "="*80)
    print(" MODE: TRAINING SAND MINING DETECTION MODEL")
    print("="*80 + "\n")

    if not os.path.exists(args.shapefile):
        print(f"❌ Error: Shapefile not found at '{args.shapefile}'")
        return None, None # Ensure tuple return on failure

    try:
        detector = SandMiningDetection()

        # Check if labels file exists and ask for confirmation if not --yes
        if os.path.exists(detector.labels_file) and not args.yes:
             # Use Tkinter for confirmation only if available, fallback to console prompt
             try:
                 root = tk.Tk()
                 root.withdraw() # Hide main window
                 proceed = messagebox.askyesno("Confirm Overwrite", f"Labels file '{detector.labels_file}' already exists.\n\nContinuing training might overwrite or add to existing labels during the labeling step.\n\nDo you want to proceed?")
                 root.destroy()
                 if not proceed:
                      print("Training cancelled by user.")
                      return None, None
             except tk.TclError: # Handle environments without display
                 confirm = input(f"Labels file '{detector.labels_file}' already exists. Proceed and potentially modify labels? (y/N): ").lower()
                 if confirm != 'y':
                      print("Training cancelled by user.")
                      return None, None


        # 1. Load shapefile and get sampling points for training
        training_coordinates = detector.load_shapefile_and_get_points(
            args.shapefile,
            sample_size=args.sample_size,
            distance_km=args.distance # Use distance for min separation
        )
        if not training_coordinates:
            print("❌ Error: Failed to get coordinates from shapefile. Cannot proceed.")
            return None, None # Ensure tuple return

        # 2. Download images for these points
        download_success = detector.download_training_images(training_coordinates)
        if not download_success:
             print("❌ Error: Failed to download sufficient training images. Cannot proceed with labeling or training.")
             return None, None # Ensure tuple return

        # 3. Label images (GUI) - Only if download was successful
        detector.label_images() # This blocks until the GUI is closed

        # 4. Train model - Only if labeling was initiated (implies images exist)
        trained_model = detector.train_model()

        if trained_model:
            print("\n✅ Training workflow completed successfully!")
            print(f"   Model saved to: {detector.model_file}")
            print(f"   Scaler saved to: {detector.scaler_file}")
            return detector.model_file, detector.scaler_file # Return paths for 'both' mode
        else:
            print("\n❌ Training workflow failed or was incomplete (check logs for labeling/training errors).")
            return None, None # Indicate failure

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the training process: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return None, None # Indicate failure

def run_mapping(args, model_path=None, scaler_path=None):
    """Run the probability mapping workflow."""
    print("\n" + "="*80)
    print(" MODE: GENERATING SAND MINING PROBABILITY MAP")
    print("="*80 + "\n")

    if not os.path.exists(args.shapefile):
        print(f"❌ Error: Shapefile not found at '{args.shapefile}'")
        return False # Indicate failure

    # Determine model and scaler paths
    model_to_use = model_path if model_path else args.model
    scaler_to_use = scaler_path if scaler_path else args.scaler

    # Check if model and scaler exist before initializing mapper
    if not os.path.exists(model_to_use):
         print(f"❌ Error: Model file not found: '{model_to_use}'")
         print("   Please train a model first (use --mode train or --mode both) or provide a valid path using --model.")
         return False
    if not os.path.exists(scaler_to_use):
         print(f"❌ Error: Scaler file not found: '{scaler_to_use}'")
         print("   This file should have been created during training. Ensure training completed successfully.")
         return False

    try:
        # Initialize mapper with specified model and scaler
        mapper = SandMiningProbabilityMapper(model_path=model_to_use, scaler_path=scaler_to_use)

        # Generate probability map
        results_df = mapper.create_probability_map(
            shapefile_path=args.shapefile,
            distance_km=args.distance,
            output_file=args.output # Pass the user-specified output filename
        )

        if results_df is not None and not results_df.empty:
            print("\n✅ Mapping workflow completed successfully!")

            # Find the actual date range from results for the report
            try:
                min_img_date = results_df['image_date'].min()
                max_img_date = results_df['image_date'].max()
                date_range_str = f"{min_img_date} to {max_img_date}" if min_img_date != max_img_date else min_img_date
            except Exception:
                 date_range_str = "Unknown" # Fallback if error getting dates


            # Generate report (summary plot)
            report_file = mapper.create_report(
                results_df,
                args.shapefile,
                date_range_str, # Pass the actual date range
                output_file=None # Use default filename for plot
            )
            if report_file:
                print(f"   Summary report plot generated: {report_file}")
            return True # Indicate success
        else:
            print("\n❌ Mapping workflow failed or produced no results.")
            return False # Indicate failure

    except FileNotFoundError as fnf_err:
         print(f"❌ Error: {fnf_err}") # Handle file not found errors during mapping init or processing
         return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during mapping: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return False # Indicate failure

# --- Main Execution Block ---
def main():
    # Import sklearn here specifically for the version check
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "Not Found"
    # Get other versions safely
    try: gee_version = ee.__version__
    except Exception: gee_version = "Not Found"
    try: pd_version = pd.__version__
    except Exception: pd_version = "Not Found"
    try: gpd_version = gpd.__version__
    except Exception: gpd_version = "Not Found"


    print(
        f"""
==========================================================
     SAND MINING DETECTION AND MAPPING TOOL v1.4
==========================================================
 Using Google Earth Engine and Machine Learning to identify
 potential sand mining activities along rivers.
----------------------------------------------------------
 Python version: {sys.version.split()[0]}
 GEE API version: {gee_version}
 Pandas version: {pd_version}
 Geopandas version: {gpd_version}
 Scikit-learn version: {sklearn_version}
----------------------------------------------------------"""
    )
    args = parse_arguments()

    # Initial check for shapefile existence
    if not os.path.exists(args.shapefile):
        print(f"❌ FATAL ERROR: Input shapefile not found at the specified path:")
        print(f"   '{args.shapefile}'")
        print("   Please provide the correct path to your .shp file.")
        sys.exit(1) # Exit immediately if shapefile is missing


    model_path_from_train = None
    scaler_path_from_train = None
    training_success = False
    mapping_success = False


    # --- Execute based on mode ---
    if args.mode == 'train' or args.mode == 'both':
        # run_training now returns (model_path, scaler_path) or (None, None)
        model_path_from_train, scaler_path_from_train = run_training(args)
        # Training is successful if both paths are returned (not None)
        training_success = model_path_from_train is not None and scaler_path_from_train is not None

    if args.mode == 'map' or (args.mode == 'both' and training_success):
        if args.mode == 'both':
             print("\n----------------------------------------------------------")
             print(" Proceeding to mapping using the model just trained...")
             print("----------------------------------------------------------")
             # Pass the paths obtained from training
             mapping_success = run_mapping(args, model_path=model_path_from_train, scaler_path=scaler_path_from_train)
        else: # mode == 'map'
             # run_mapping handles checking for args.model/args.scaler existence
             mapping_success = run_mapping(args) # Uses paths from args or defaults

    elif args.mode == 'both' and not training_success:
         print("\n❌ Training failed. Skipping mapping step.")


    # --- Final Summary ---
    print("\n" + "="*80)
    print(" TOOL EXECUTION SUMMARY")
    print("="*80)
    final_status = 0 # 0 for success, 1 for failure

    if args.mode == 'train' or args.mode == 'both':
        status_msg = '✅ SUCCESS' if training_success else '❌ FAILED'
        print(f"Training attempt: {status_msg}")
        if not training_success: final_status = 1

    if args.mode == 'map' or args.mode == 'both':
         # Check if mapping was skipped due to training failure
         if args.mode == 'both' and not training_success:
              print("Mapping attempt:  SKIPPED due to training failure")
              # Don't mark final status as fail if only mapping was skipped due to prior train fail
         else:
              status_msg = '✅ SUCCESS' if mapping_success else '❌ FAILED'
              print(f"Mapping attempt:  {status_msg}")
              if not mapping_success: final_status = 1 # Mark fail if mapping itself failed
    print("="*80)

    if final_status != 0:
        print("\nOne or more critical steps failed. Please review the logs above for errors.")
    else:
        print("\nTool finished successfully.")

    sys.exit(final_status) # Exit with appropriate code


if __name__ == "__main__":
    # Ensure necessary folders exist before anything else
    os.makedirs(BASE_FOLDER, exist_ok=True)
    os.makedirs(RIVER_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(PROBABILITY_MAPS_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    main()
