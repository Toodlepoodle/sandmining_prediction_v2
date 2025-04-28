# Simple Sand Mining Detection Workflow - Improved Version
# ======================================================

import ee
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk  # Added ImageTk import here
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import io
import time
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import geopandas as gpd
import random

class SandMiningDetection:
    def __init__(self):
        # First, handle Earth Engine authentication properly
        try:
            ee.Initialize()
            print("Successfully initialized Earth Engine")
        except:
            print("Need to authenticate with Earth Engine...")
            print("\nSTEP 1: First, make sure you have signed up for Google Earth Engine at:")
            print("https://signup.earthengine.google.com/")
            print("\nSTEP 2: After signing up, run this authentication process...")
            
            try:
                # Attempt authentication
                ee.Authenticate(auth_mode='localhost')  # or 'notebook' if that doesn't work
                print("Authentication successful!")
                
                # Try to initialize with your project
                try:
                    ee.Initialize()
                    print("Successfully initialized Earth Engine")
                except:
                    # If initialization fails, ask for project ID
                    project_id = input("\nEnter your Google Cloud Project ID (you can find this in your Google Cloud Console): ")
                    if project_id:
                        ee.Initialize(project=project_id)
                        print("Successfully initialized Earth Engine with project ID")
                    else:
                        print("Project ID is required. Please create a project in Google Cloud Console.")
                        raise Exception("No project ID provided")
            except Exception as e:
                print(f"\nAuthentication failed: {str(e)}")
                print("\nPlease follow these steps:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project or select an existing one")
                print("3. Enable the Earth Engine API for your project")
                print("4. Run this script again")
                raise e
        
        # Create folders
        self.base_folder = 'sand_mining_detection'
        self.yes_folder = os.path.join(self.base_folder, 'YES_SANDMINING')
        self.no_folder = os.path.join(self.base_folder, 'NO_SANDMINING')
        self.temp_folder = os.path.join(self.base_folder, 'temp')
        
        os.makedirs(self.yes_folder, exist_ok=True)
        os.makedirs(self.no_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        
        print(f"Created folders at: {self.base_folder}")
    
    def load_shapefile_and_get_points(self, shapefile_path, sample_size=20, distance_km=5):
        """Load shapefile and sample points along the river at regular intervals"""
        print(f"Loading shapefile from: {shapefile_path}")
        
        try:
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Get the river geometry
            if isinstance(gdf.geometry.iloc[0], gpd.geoseries.GeoSeries):
                river_geom = gdf.geometry.iloc[0]
            else:
                river_geom = gdf.geometry.union_all()
            
            coordinates = []
            
            # If the geometry is a MultiLineString or MultiPolygon, convert to single geometry
            if river_geom.geom_type == 'MultiLineString':
                # Convert MultiLineString to single LineString by taking the longest segment
                lines = list(river_geom.geoms)
                river_geom = max(lines, key=lambda x: x.length)
            elif river_geom.geom_type == 'MultiPolygon':
                # Convert MultiPolygon to single Polygon by taking the largest
                polygons = list(river_geom.geoms)
                river_geom = max(polygons, key=lambda x: x.area)
            
            # Extract coordinates based on geometry type
            if river_geom.geom_type == 'LineString':
                coords = list(river_geom.coords)
            elif river_geom.geom_type == 'Polygon':
                coords = list(river_geom.exterior.coords)
            else:
                print(f"Unexpected geometry type: {river_geom.geom_type}")
                return None
            
            # Convert distance to degrees (approximation: 1 degree ≈ 111 km)
            distance_degrees = distance_km / 111.0
            
            # Calculate total length
            total_length = 0
            for i in range(len(coords) - 1):
                segment_length = ((coords[i+1][0] - coords[i][0])**2 + 
                                  (coords[i+1][1] - coords[i][1])**2)**0.5
                total_length += segment_length
            
            # Calculate number of points that can fit
            n_points = int(total_length / distance_degrees)
            if n_points < sample_size:
                print(f"River too short for {sample_size} points at {distance_km}km spacing.")
                print(f"Maximum possible points: {n_points}")
                sample_size = min(sample_size, n_points)
            
            # Sample points along the river
            current_distance = 0
            target_distance = 0
            point_index = 0
            
            for i in range(len(coords) - 1):
                start_coord = coords[i]
                end_coord = coords[i + 1]
                
                segment_length = ((end_coord[0] - start_coord[0])**2 + 
                                  (end_coord[1] - start_coord[1])**2)**0.5
                
                while current_distance + segment_length >= target_distance and point_index < sample_size:
                    # Interpolate point along the segment
                    t = (target_distance - current_distance) / segment_length
                    lon = start_coord[0] + t * (end_coord[0] - start_coord[0])
                    lat = start_coord[1] + t * (end_coord[1] - start_coord[1])
                    
                    coordinates.append((lat, lon))
                    point_index += 1
                    target_distance += distance_degrees
                
                current_distance += segment_length
            
            print(f"Generated {len(coordinates)} sampling points along the river")
            return coordinates
        
        except Exception as e:
            print(f"Error loading shapefile: {str(e)}")
            return None
    
    def download_images(self, coordinates):
        """Download images for given coordinates"""
        print(f"Downloading images for {len(coordinates)} locations...")
        
        for i, (lat, lon) in enumerate(coordinates):
            try:
                print(f"Downloading location {i+1}/{len(coordinates)}: {lat}, {lon}")
                
                # Create a point and LARGER buffer for better visibility
                point = ee.Geometry.Point([lon, lat])
                region = point.buffer(1000)  # Increased to 1000m radius = 2km patch
                
                # Load Sentinel-2 imagery with improved parameters
                s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate('2022-01-01', '2022-12-31') \
                    .filterBounds(region) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                    .select(['B2', 'B3', 'B4', 'B8']) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE') \
                    .first()  # Get the least cloudy image instead of median
                
                # Enhanced visualization parameters
                rgb_url = s2.getThumbURL({
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.4,  # Enhance visibility
                    'bands': ['B4', 'B3', 'B2'],  # RGB bands
                    'dimensions': 800,  # Larger image
                    'region': region,
                    'format': 'png'
                })
                
                # Download and save to temp folder
                response = requests.get(rgb_url)
                img = Image.open(io.BytesIO(response.content))
                filename = f'location_{i+1}_lat{lat}_lon{lon}.png'
                img.save(os.path.join(self.temp_folder, filename))
                
                time.sleep(1)  # Avoid rate limits
                print(f"Successfully downloaded image {i+1}")
            except Exception as e:
                print(f"Error downloading image for {lat}, {lon}: {str(e)}")
                continue
        
        print("All images downloaded!")
    
    def label_images(self):
        """Open GUI for labeling images"""
        
        class SimpleLabelingGUI:
            def __init__(self, temp_folder, yes_folder, no_folder):
                self.temp_folder = temp_folder
                self.yes_folder = yes_folder
                self.no_folder = no_folder
                self.images = [f for f in os.listdir(temp_folder) if f.endswith('.png')]
                self.current_index = 0
                
                # Create main window
                self.root = tk.Tk()
                self.root.title("Sand Mining Image Labeler")
                
                # Make window full screen
                self.root.attributes('-fullscreen', True)
                
                # Exit fullscreen with ESC key
                self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
                
                # Main container
                self.main_frame = tk.Frame(self.root)
                self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
                
                # Progress label
                self.progress_label = tk.Label(self.main_frame, 
                                             text="", 
                                             font=("Arial", 20, "bold"))
                self.progress_label.pack(pady=10)
                
                # Image canvas
                self.canvas = tk.Canvas(self.main_frame, bg='white')
                self.canvas.pack(pady=10, fill="both", expand=True)
                
                # Filename label
                self.filename_label = tk.Label(self.main_frame, 
                                             text="", 
                                             font=("Arial", 14))
                self.filename_label.pack(pady=5)
                
                # Button frame at the bottom
                button_frame = tk.Frame(self.main_frame)
                button_frame.pack(side=tk.BOTTOM, pady=30, fill="x")
                
                # Simple buttons without complex instructions
                self.yes_button = tk.Button(button_frame, 
                                          text="YES - Sand Mining", 
                                          command=lambda: self.label_image(True),
                                          font=("Arial", 24, "bold"),
                                          width=25,
                                          height=2,
                                          bg='green',
                                          fg='white')
                self.yes_button.pack(side="left", padx=30, expand=True)
                
                self.no_button = tk.Button(button_frame, 
                                         text="NO - No Sand Mining", 
                                         command=lambda: self.label_image(False),
                                         font=("Arial", 24, "bold"),
                                         width=25,
                                         height=2,
                                         bg='red',
                                         fg='white')
                self.no_button.pack(side="left", padx=30, expand=True)
                
                self.skip_button = tk.Button(button_frame, 
                                           text="Skip / Unclear", 
                                           command=self.skip_image,
                                           font=("Arial", 24, "bold"),
                                           width=25,
                                           height=2,
                                           bg='gray',
                                           fg='white')
                self.skip_button.pack(side="left", padx=30, expand=True)
                
                # Load first image
                self.root.after(100, self.load_current_image)  # Delayed loading to ensure window is ready
                
                # Start event loop
                self.root.mainloop()
            
            def load_current_image(self):
                if self.current_index >= len(self.images):
                    messagebox.showinfo("Complete", "All images labeled!")
                    self.root.destroy()
                    return
                
                # Update progress
                progress_text = f"Image {self.current_index + 1} of {len(self.images)}"
                self.progress_label.config(text=progress_text)
                
                # Load image
                current_image = self.images[self.current_index]
                self.filename_label.config(text=current_image)
                
                img_path = os.path.join(self.temp_folder, current_image)
                image = Image.open(img_path)
                
                # Get screen and canvas dimensions
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                # Handle initial canvas size
                if canvas_width <= 1:
                    canvas_width = self.root.winfo_screenwidth() - 40
                if canvas_height <= 1:
                    canvas_height = self.root.winfo_screenheight() - 300
                
                # Resize image to fit canvas while maintaining aspect ratio
                image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(image)
                
                # Display on canvas
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
            
            def label_image(self, has_sand_mining):
                current_image = self.images[self.current_index]
                src_path = os.path.join(self.temp_folder, current_image)
                
                # Move to appropriate folder
                if has_sand_mining:
                    dest_path = os.path.join(self.yes_folder, current_image)
                else:
                    dest_path = os.path.join(self.no_folder, current_image)
                
                shutil.move(src_path, dest_path)
                
                # Next image
                self.current_index += 1
                self.load_current_image()
            
            def skip_image(self):
                """Skip unclear images"""
                self.current_index += 1
                self.load_current_image()
        
        # Run labeler
        SimpleLabelingGUI(self.temp_folder, self.yes_folder, self.no_folder)
    
    def train_model(self):
        """Train model using labeled images"""
        print("Training model...")
        
        X = []
        y = []
        
        # Process YES images
        yes_images = [f for f in os.listdir(self.yes_folder) if f.endswith('.png')]
        for img_name in yes_images:
            img_path = os.path.join(self.yes_folder, img_name)
            features = self.extract_features(img_path)
            X.append(features)
            y.append(1)
        
        # Process NO images
        no_images = [f for f in os.listdir(self.no_folder) if f.endswith('.png')]
        for img_name in no_images:
            img_path = os.path.join(self.no_folder, img_name)
            features = self.extract_features(img_path)
            X.append(features)
            y.append(0)
        
        if len(X) == 0:
            print("No labeled images found!")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training with {len(X)} images...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(self.base_folder, 'sand_mining_model.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        return model
    
    def extract_features(self, image_path):
        """Extract features from an image"""
        img = Image.open(image_path)
        img_array = np.array(img) / 255.0
        
        features = []
        
        # Color statistics for each channel
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        # Simple texture features
        gray = np.mean(img_array, axis=2)
        features.extend([
            np.std(gray),
            np.mean(np.abs(np.diff(gray, axis=0))),
            np.mean(np.abs(np.diff(gray, axis=1)))
        ])
        
        return features
    
    def predict_new_location(self, lat, lon):
        """Predict sand mining at new location"""
        # Load model
        model_path = os.path.join(self.base_folder, 'sand_mining_model.joblib')
        if not os.path.exists(model_path):
            print("Model not found! Train the model first.")
            return None
        
        model = joblib.load(model_path)
        
        # Download image for new location
        print(f"Downloading image for {lat}, {lon}...")
        
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(1000)  # Same buffer as training
            
            s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterDate('2022-01-01', '2022-12-31') \
                .filterBounds(region) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE') \
                .first()
            
            rgb_url = s2.getThumbURL({
                'min': 0,
                'max': 3000,
                'gamma': 1.4,
                'bands': ['B4', 'B3', 'B2'],
                'dimensions': 800,
                'region': region,
                'format': 'png'
            })
            
            # Download and save temporarily
            response = requests.get(rgb_url)
            img = Image.open(io.BytesIO(response.content))
            temp_path = os.path.join(self.temp_folder, 'temp_predict.png')
            img.save(temp_path)
            
            # Extract features and predict
            features = self.extract_features(temp_path)
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            result = "SAND MINING DETECTED" if prediction == 1 else "NO SAND MINING"
            confidence = probability[prediction]
            
            print(f"\nPrediction: {result}")
            print(f"Confidence: {confidence:.2f}")
            
            return prediction, confidence
        except Exception as e:
            print(f"Error predicting for {lat}, {lon}: {str(e)}")
            return None

# Main workflow
def main():
    try:
        detector = SandMiningDetection()
        
        # 1. Load shapefile and get sampling points
        shapefile_path = "DATA/DAMODAR_SHAPEFILE/mrb_basins.shp"
        # Sample points every 5km along the river
        training_coordinates = detector.load_shapefile_and_get_points(shapefile_path, sample_size=20, distance_km=5)
        
        if not training_coordinates:
            print("Failed to get coordinates from shapefile!")
            return
        
        # 2. Download images
        detector.download_images(training_coordinates)
        
        # 3. Label images (GUI opens automatically)
        detector.label_images()
        
        # 4. Train model
        model = detector.train_model()
        
        # 5. Make predictions on new locations
        if model:
            new_location = (23.65, 86.85)  # Example new location
            result = detector.predict_new_location(*new_location)
            if result:
                prediction, confidence = result
                print(f"Done! Prediction: {prediction}, Confidence: {confidence}")
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have:")
        print("1. Signed up for Google Earth Engine")
        print("2. Created a Google Cloud Project")
        print("3. Enabled the Earth Engine API")
        print("4. Authenticated properly")

if __name__ == "__main__":
    main()