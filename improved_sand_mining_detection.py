"""
Improved Sand Mining Detection with Better Image Quality
========================================================
This version uses higher resolution imagery and enhanced visualization techniques.
"""

import ee
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
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
        """Download images for given coordinates using higher quality satellite sources"""
        print(f"Downloading high-resolution images for {len(coordinates)} locations...")
        
        for i, (lat, lon) in enumerate(coordinates):
            try:
                print(f"Downloading location {i+1}/{len(coordinates)}: {lat}, {lon}")
                
                # Create a point and use a MUCH LARGER buffer to ensure the river is visible
                point = ee.Geometry.Point([lon, lat])
                region = point.buffer(2500)  # 2500m radius = 5km patch to capture more of the river
                
                # Try to find the river course within the buffer to better center the image
                # This helps ensure the river is visible in the downloaded image
                try:
                    # Calculate NDWI (Normalized Difference Water Index) to find water bodies
                    def find_river_in_region(region):
                        # Get recent clear imagery 
                        recent_img = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                            .filterBounds(region) \
                            .filterDate(ee.Date.now().advance(-6, 'month'), ee.Date.now()) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                            .sort('CLOUDY_PIXEL_PERCENTAGE') \
                            .first()
                        
                        if recent_img is None:
                            return region  # Return original region if no image found
                        
                        # Calculate NDWI (Green-NIR)/(Green+NIR)
                        ndwi = recent_img.normalizedDifference(['B3', 'B8'])
                        
                        # Threshold NDWI to extract water (values > 0.3 are likely water)
                        water = ndwi.gt(0.2)
                        
                        # Try to find water pixels and refine the region
                        water_vectors = water.selfMask().reduceToVectors(
                            geometry=region,
                            scale=10,
                            geometryType='polygon',
                            eightConnected=True,
                            maxPixels=1e9
                        )
                        
                        # If water bodies found, use the largest one (likely the river)
                        if water_vectors.size().getInfo() > 0:
                            # Get the largest water body (likely the main river)
                            largest_water = water_vectors.sort('area', False).first()
                            water_centroid = largest_water.centroid(10)
                            
                            # Create a new region centered on the water body but maintain the original size
                            return water_centroid.buffer(2500)
                        else:
                            return region  # Return original region if no water found
                    
                    # Try to refine the region to center on the river
                    region = find_river_in_region(region)
                except Exception as e:
                    print(f"River finding failed (using original region): {str(e)}")
                    # Continue with the original region
                
                # ==========================================
                # METHOD 1: Using NICFI Planet high-resolution imagery (approx 5m resolution)
                # Available in Earth Engine only for researches with proper access
                try:
                    # Try to access NICFI Planet data (requires special access)
                    planet = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/asia') \
                        .filterDate(ee.Date('2022-01-01'), ee.Date('2022-12-31')) \
                        .sort('system:time_start', False) \
                        .first()
                    
                    if planet:
                        planet_url = planet.getThumbURL({
                            'bands': ['R', 'G', 'B'],
                            'min': 0,
                            'max': 2500,
                            'gamma': 1.2,
                            'dimensions': 1024,  # Higher resolution
                            'region': region,
                            'format': 'png'
                        })
                        
                        # Download and save as Planet high-res image
                        response = requests.get(planet_url)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            filename = f'planet_location_{i+1}_lat{lat}_lon{lon}.png'
                            img.save(os.path.join(self.temp_folder, filename))
                            print(f"Successfully downloaded high-res Planet image {i+1}")
                            continue  # Skip to next coordinate if Planet worked
                except Exception as e:
                    print(f"Planet NICFI access failed (requires special access): {str(e)}")
                    # Continue to alternate sources
                
                # ==========================================
                # METHOD 2: Using Sentinel-2 Harmonized imagery (10m resolution but better quality)
                try:
                    # Get most recent least cloudy image
                    s2_harmonized = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                        .filterDate('2022-01-01', '2022-12-31') \
                        .filterBounds(region) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                        .sort('CLOUDY_PIXEL_PERCENTAGE') \
                        .first()
                    
                    # If no good Sentinel-2 image, try another time range
                    if not s2_harmonized:
                        s2_harmonized = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                            .filterDate('2021-01-01', '2022-12-31') \
                            .filterBounds(region) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)) \
                            .sort('CLOUDY_PIXEL_PERCENTAGE') \
                            .first()
                    
                    if s2_harmonized:
                        # Apply NDWI for better water visibility (helps with sand mining detection)
                        ndwi = s2_harmonized.normalizedDifference(['B3', 'B8']).rename('NDWI')
                        
                        # Create false color composite for better feature distinction
                        composite = s2_harmonized.addBands(ndwi)
                        
                        # Enhanced visualization with higher dimensions
                        rgb_url = composite.getThumbURL({
                            'min': 0,
                            'max': 3000,
                            'gamma': 1.3,
                            'bands': ['B4', 'B3', 'B2'],  # True color
                            'dimensions': 1024,  # Larger image
                            'region': region,
                            'format': 'png'
                        })
                        
                        # Also get a false color composite specifically designed to highlight water bodies and river courses
                        falsecolor_url = composite.getThumbURL({
                            'bands': ['B8', 'B11', 'B4'],  # Modified false color that highlights water better
                            'min': 0,
                            'max': 3500,
                            'gamma': 1.6,  # Higher gamma for better contrast
                            'dimensions': 2048,  # Much higher resolution
                            'region': region,
                            'format': 'png'
                        })
                        
                        # Download both types of images
                        # True color
                        response = requests.get(rgb_url)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            # Enhance image quality using PIL
                            enhancer = ImageEnhance.Contrast(img)
                            img = enhancer.enhance(1.2)  # Increase contrast
                            enhancer = ImageEnhance.Sharpness(img)
                            img = enhancer.enhance(1.3)  # Increase sharpness
                            
                            filename = f'location_{i+1}_lat{lat}_lon{lon}.png'
                            img.save(os.path.join(self.temp_folder, filename))
                            
                            # False color - save as separate file for reference
                            response_false = requests.get(falsecolor_url)
                            if response_false.status_code == 200:
                                img_false = Image.open(io.BytesIO(response_false.content))
                                filename_false = f'falsecolor_location_{i+1}_lat{lat}_lon{lon}.png'
                                img_false.save(os.path.join(self.temp_folder, filename_false))
                            
                            print(f"Successfully downloaded enhanced Sentinel-2 image {i+1}")
                            continue  # Skip to next coordinate
                except Exception as e:
                    print(f"Sentinel-2 error: {str(e)}")
                
                # ==========================================
                # METHOD 3: Fallback to standard Sentinel-2 but with better processing
                try:
                    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                        .filterDate('2022-01-01', '2022-12-31') \
                        .filterBounds(region) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
                        .sort('CLOUDY_PIXEL_PERCENTAGE') \
                        .first()
                    
                    # Create indices to help with visualization
                    ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
                    
                                            # Enhanced visualization parameters with better contrast for river visibility
                        rgb_url = s2.getThumbURL({
                            'min': 0,
                            'max': 2500,  # Lower max value to increase contrast
                            'gamma': 1.5,  # Higher gamma for better contrast
                            'bands': ['B4', 'B3', 'B2'],
                            'dimensions': 2048,  # Much higher resolution to reduce graininess
                            'region': region,
                            'format': 'png'
                        })
                    
                    # Download and save to temp folder
                    response = requests.get(rgb_url)
                    img = Image.open(io.BytesIO(response.content))
                    
                    # Post-process the image to improve quality
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.3)  # Increase contrast
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(1.4)  # Increase sharpness
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.1)  # Slightly increase brightness
                    
                    filename = f'location_{i+1}_lat{lat}_lon{lon}.png'
                    img.save(os.path.join(self.temp_folder, filename))
                    
                    print(f"Successfully downloaded fallback enhanced image {i+1}")
                except Exception as e:
                    print(f"Fallback image error: {str(e)}")
                    continue
                
                time.sleep(1)  # Avoid rate limits
            except Exception as e:
                print(f"Error downloading image for {lat}, {lon}: {str(e)}")
                continue
        
        print("All images downloaded!")

    def label_images(self):
        """Open GUI for labeling images with enhanced viewer"""
        
        class EnhancedLabelingGUI:
            def __init__(self, temp_folder, yes_folder, no_folder):
                self.temp_folder = temp_folder
                self.yes_folder = yes_folder
                self.no_folder = no_folder
                self.images = [f for f in os.listdir(temp_folder) if f.endswith('.png') and not f.startswith('falsecolor_')]
                self.current_index = 0
                self.zoom_level = 1.0
                
                # Create main window
                self.root = tk.Tk()
                self.root.title("Enhanced Sand Mining Image Labeler")
                
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
                
                # Image canvas with scrollbars
                self.canvas_frame = tk.Frame(self.main_frame)
                self.canvas_frame.pack(fill="both", expand=True, pady=10)
                
                # Add scrollbars
                h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
                h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
                
                v_scrollbar = tk.Scrollbar(self.canvas_frame)
                v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Canvas for the image
                self.canvas = tk.Canvas(self.canvas_frame, bg='white',
                                       xscrollcommand=h_scrollbar.set,
                                       yscrollcommand=v_scrollbar.set)
                self.canvas.pack(side=tk.LEFT, fill="both", expand=True)
                
                # Configure scrollbars
                h_scrollbar.config(command=self.canvas.xview)
                v_scrollbar.config(command=self.canvas.yview)
                
                # Zoom controls
                zoom_frame = tk.Frame(self.main_frame)
                zoom_frame.pack(pady=5)
                
                zoom_in_btn = tk.Button(zoom_frame, text="Zoom In (+)", command=self.zoom_in)
                zoom_in_btn.pack(side=tk.LEFT, padx=5)
                
                zoom_out_btn = tk.Button(zoom_frame, text="Zoom Out (-)", command=self.zoom_out)
                zoom_out_btn.pack(side=tk.LEFT, padx=5)
                
                reset_zoom_btn = tk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom)
                reset_zoom_btn.pack(side=tk.LEFT, padx=5)
                
                # Toggle false color button 
                self.show_false_color = False
                self.toggle_color_btn = tk.Button(zoom_frame, text="Toggle False Color", command=self.toggle_false_color)
                self.toggle_color_btn.pack(side=tk.LEFT, padx=20)
                
                # Mouse wheel binding for zoom
                self.canvas.bind("<MouseWheel>", self.mouse_wheel)  # Windows
                self.canvas.bind("<Button-4>", self.mouse_wheel)  # Linux scroll up
                self.canvas.bind("<Button-5>", self.mouse_wheel)  # Linux scroll down
                
                # Filename label
                self.filename_label = tk.Label(self.main_frame, 
                                             text="", 
                                             font=("Arial", 14))
                self.filename_label.pack(pady=5)
                
                # Help text
                help_text = "Look for: Disturbed water patterns, barges, exposed sand banks, excavation sites"
                self.help_label = tk.Label(self.main_frame, text=help_text, font=("Arial", 12))
                self.help_label.pack(pady=5)
                
                # Button frame at the bottom
                button_frame = tk.Frame(self.main_frame)
                button_frame.pack(side=tk.BOTTOM, pady=30, fill="x")
                
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
                self.root.after(100, self.load_current_image)  # Delayed loading
                
                # Start event loop
                self.root.mainloop()
            
            def zoom_in(self):
                self.zoom_level *= 1.2
                self.load_current_image()
            
            def zoom_out(self):
                self.zoom_level /= 1.2
                self.load_current_image()
            
            def reset_zoom(self):
                self.zoom_level = 1.0
                self.load_current_image()
            
            def mouse_wheel(self, event):
                # Zoom with mouse wheel
                if event.num == 4 or event.delta > 0:
                    self.zoom_in()
                elif event.num == 5 or event.delta < 0:
                    self.zoom_out()
            
            def toggle_false_color(self):
                self.show_false_color = not self.show_false_color
                self.load_current_image()
            
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
                
                # Determine which image to show (normal or false color)
                if self.show_false_color:
                    # Try to load false color version
                    false_name = 'falsecolor_' + current_image
                    false_path = os.path.join(self.temp_folder, false_name)
                    
                    if os.path.exists(false_path):
                        img_path = false_path
                        self.toggle_color_btn.config(text="Show Normal Color", bg='orange')
                    else:
                        img_path = os.path.join(self.temp_folder, current_image)
                        self.toggle_color_btn.config(text="False Color Not Available", bg='gray')
                else:
                    img_path = os.path.join(self.temp_folder, current_image)
                    self.toggle_color_btn.config(text="Show False Color", bg='lightblue')
                
                image = Image.open(img_path)
                
                # Apply zoom
                width, height = image.size
                new_width = int(width * self.zoom_level)
                new_height = int(height * self.zoom_level)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            
            def label_image(self, has_sand_mining):
                current_image = self.images[self.current_index]
                src_path = os.path.join(self.temp_folder, current_image)
                
                # Move to appropriate folder
                if has_sand_mining:
                    dest_path = os.path.join(self.yes_folder, current_image)
                else:
                    dest_path = os.path.join(self.no_folder, current_image)
                
                # Copy the image instead of moving it to keep false color reference
                shutil.copy(src_path, dest_path)
                
                # Next image
                self.current_index += 1
                self.load_current_image()
            
            def skip_image(self):
                """Skip unclear images"""
                self.current_index += 1
                self.load_current_image()
        
        # Run enhanced labeler
        EnhancedLabelingGUI(self.temp_folder, self.yes_folder, self.no_folder)
    
    def train_model(self):
        """Train model using labeled images with improved features"""
        print("Training model with enhanced features...")
        
        X = []
        y = []
        
        # Process YES images
        yes_images = [f for f in os.listdir(self.yes_folder) if f.endswith('.png')]
        for img_name in yes_images:
            img_path = os.path.join(self.yes_folder, img_name)
            features = self.extract_enhanced_features(img_path)
            X.append(features)
            y.append(1)
        
        # Process NO images
        no_images = [f for f in os.listdir(self.no_folder) if f.endswith('.png')]
        for img_name in no_images:
            img_path = os.path.join(self.no_folder, img_name)
            features = self.extract_enhanced_features(img_path)
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
        
        # Train model with more estimators and better hyperparameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = model.feature_importances_
        print("\nTop 10 Feature Importances:")
        top_indices = np.argsort(feature_importance)[-10:]
        for i in top_indices:
            print(f"Feature {i}: {feature_importance[i]:.4f}")
        
        # Save model
        model_path = os.path.join(self.base_folder, 'sand_mining_model.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        return model
    
    def extract_enhanced_features(self, image_path):
        """Extract enhanced features from an image"""
        img = Image.open(image_path)
        img_array = np.array(img) / 255.0
        
        features = []
        
        # Basic color statistics for each channel
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
                np.max(channel_data),
                np.min(channel_data)
            ])
        
        # Create grayscale image
        gray = np.mean(img_array, axis=2)
        
        # Advanced texture features
        features.extend([
            np.std(gray),
            np.mean(np.abs(np.diff(gray, axis=0))),  # Horizontal gradient
            np.mean(np.abs(np.diff(gray, axis=1))),  # Vertical gradient
            np.std(np.abs(np.diff(gray, axis=0))),   # Gradient variance
            np.std(np.abs(np.diff(gray, axis=1)))    # Gradient variance
        ])
        
        # Image is divided into 4x4 grid for localized features
        h, w = gray.shape
        h_step = h // 4
        w_step = w // 4
        
        for i in range(4):
            for j in range(4):
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                
                # Get region
                region = gray[h_start:h_end, w_start:w_end]
                
                # Region features
                features.extend([
                    np.mean(region),
                    np.std(region)
                ])
                
                # Color features for each channel in the region
                for channel in range(3):
                    region_color = img_array[h_start:h_end, w_start:w_end, channel]
                    features.append(np.mean(region_color))
        
        # Calculate image entropy (measure of texture complexity)
        hist, _ = np.histogram(gray, bins=32, range=(0, 1))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(entropy)
        
        # Water detection features (key for sand mining detection)
        # Calculate band ratios that help identify water and disturbed patterns
        blue_green_ratio = np.mean(img_array[:,:,2]) / (np.mean(img_array[:,:,1]) + 1e-10)
        features.append(blue_green_ratio)
        
        # Calculate approximate NDWI using blue/green channels
        # (Not true NDWI without NIR, but approximation)
        pseudo_ndwi = (np.mean(img_array[:,:,1]) - np.mean(img_array[:,:,0])) / \
                      (np.mean(img_array[:,:,1]) + np.mean(img_array[:,:,0]) + 1e-10)
        features.append(pseudo_ndwi)
        
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
            
            # Try to get high-quality Sentinel-2 image
            s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterDate('2022-01-01', '2022-12-31') \
                .filterBounds(region) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE') \
                .first()
            
            if not s2:
                # Fall back to regular Sentinel-2
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
                'dimensions': 1024,
                'region': region,
                'format': 'png'
            })
            
            # Download and save temporarily
            response = requests.get(rgb_url)
            img = Image.open(io.BytesIO(response.content))
            
            # Enhance the image
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            temp_path = os.path.join(self.temp_folder, 'temp_predict.png')
            img.save(temp_path)
            
            # Extract features and predict
            features = self.extract_enhanced_features(temp_path)
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