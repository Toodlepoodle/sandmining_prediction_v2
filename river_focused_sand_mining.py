"""
River-Focused Sand Mining Detection
=================================
This script specifically focuses on ensuring the river is visible in all downloaded images.
It creates high-quality, centered images of river sections for better sand mining detection.
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
import geopandas as gpd
from shapely.geometry import mapping
import json

class RiverFocusedSandMining:
    def __init__(self):
        # Initialize Earth Engine with proper authentication
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
                ee.Authenticate(auth_mode='localhost')
                print("Authentication successful!")
                
                # Try to initialize with your project
                try:
                    ee.Initialize()
                    print("Successfully initialized Earth Engine")
                except:
                    # If initialization fails, ask for project ID
                    project_id = input("\nEnter your Google Cloud Project ID: ")
                    if project_id:
                        ee.Initialize(project=project_id)
                        print("Successfully initialized Earth Engine with project ID")
                    else:
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
        self.river_images_folder = os.path.join(self.base_folder, 'river_images')
        
        os.makedirs(self.river_images_folder, exist_ok=True)
        print(f"Created folders at: {self.base_folder}")
    
    def load_river_shapefile(self, shapefile_path):
        """Load river shapefile and prepare it for processing"""
        print(f"Loading river shapefile from: {shapefile_path}")
        
        try:
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Convert to GeoJSON for Earth Engine
            geojson = json.loads(gdf.to_json())
            
            # Create Earth Engine geometry
            river_features = []
            for feature in geojson['features']:
                geom = feature['geometry']
                if geom['type'] in ['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon']:
                    river_features.append(ee.Geometry.Polygon(mapping(gdf.geometry.iloc[0])))
            
            # Create a FeatureCollection for the river
            river_collection = ee.FeatureCollection(river_features)
            
            print(f"Successfully loaded river geometry")
            return river_collection, gdf
        
        except Exception as e:
            print(f"Error loading shapefile: {str(e)}")
            return None, None
    
    def generate_river_segments(self, gdf, distance_km=5):
        """Generate sampling points along the river centerline"""
        try:
            # Get the river geometry
            if isinstance(gdf.geometry.iloc[0], gpd.geoseries.GeoSeries):
                river_geom = gdf.geometry.iloc[0]
            else:
                river_geom = gdf.geometry.union_all()
            
            # Create a line along the river centerline
            if river_geom.geom_type == 'Polygon' or river_geom.geom_type == 'MultiPolygon':
                # For polygons, we need to extract a centerline
                # This is a simplification - in a production environment you might 
                # want to use a proper centerline extraction algorithm
                print("Converting polygon to centerline (approximate)")
                centerline = river_geom.centroid.buffer(0.0001).boundary
            else:
                # For lines, use the geometry directly
                centerline = river_geom
            
            # Sample points along the centerline
            points = []
            distance_degrees = distance_km / 111.0  # Convert km to approximate degrees
            
            if centerline.geom_type == 'LineString':
                line_length = centerline.length
                num_segments = max(10, int(line_length / distance_degrees))
                
                for i in range(num_segments + 1):
                    segment_distance = i / num_segments
                    point = centerline.interpolate(segment_distance * line_length)
                    points.append((point.y, point.x))  # (lat, lon)
            
            elif centerline.geom_type == 'MultiLineString':
                for line in centerline.geoms:
                    line_length = line.length
                    num_segments = max(3, int(line_length / distance_degrees))
                    
                    for i in range(num_segments + 1):
                        segment_distance = i / num_segments
                        point = line.interpolate(segment_distance * line_length)
                        points.append((point.y, point.x))  # (lat, lon)
            
            print(f"Generated {len(points)} river segments for analysis")
            return points
        
        except Exception as e:
            print(f"Error generating river segments: {str(e)}")
            return []
    
    def download_river_focused_images(self, shapefile_path, distance_km=5):
        """Download high-quality images centered on the river"""
        # Load the river shapefile
        river_collection, river_gdf = self.load_river_shapefile(shapefile_path)
        
        if river_gdf is None:
            print("Failed to load river shapefile")
            return False
        
        # Generate points along the river
        points = self.generate_river_segments(river_gdf, distance_km)
        
        if not points:
            print("Failed to generate river points")
            return False
        
        # Convert river to Earth Engine geometry for visualization
        river_ee = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in points
        ])
        
        # Download an image for each point
        print(f"Downloading {len(points)} high-quality river images...")
        
        for i, (lat, lon) in enumerate(points):
            try:
                print(f"Processing river segment {i+1}/{len(points)}: {lat}, {lon}")
                
                # Create a point with a buffer large enough to show river context
                point = ee.Geometry.Point([lon, lat])
                region = point.buffer(3000)  # 3km buffer
                
                # Get river geometry within this region
                river_segment = river_collection.geometry().intersection(region)
                
                # Use the latest cloud-free imagery
                images = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                    .filterDate(ee.Date.now().advance(-6, 'month'), ee.Date.now()) \
                    .filterBounds(region) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE')
                
                image_count = images.size().getInfo()
                
                if image_count == 0:
                    print("  No recent clear S2 images found, trying with longer date range")
                    images = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                        .filterDate(ee.Date.now().advance(-12, 'month'), ee.Date.now()) \
                        .filterBounds(region) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                        .sort('CLOUDY_PIXEL_PERCENTAGE')
                    
                    image_count = images.size().getInfo()
                
                if image_count > 0:
                    best_image = images.first()
                    
                    # Calculate NDWI for water detection
                    ndwi = best_image.normalizedDifference(['B3', 'B8']).rename('ndwi')
                    
                    # Add NDWI as a band and create a composite that highlights water
                    composite = best_image.addBands(ndwi)
                    
                    # Create a visualization that emphasizes the river
                    # True color with high resolution
                    rgb_url = composite.getThumbURL({
                        'bands': ['B4', 'B3', 'B2'],
                        'min': 0,
                        'max': 2000,
                        'gamma': 1.5,
                        'dimensions': 3072,  # Very high resolution
                        'region': region,
                        'format': 'png'
                    })
                    
                    # NDWI visualization to clearly show the river in blue
                    ndwi_url = composite.getThumbURL({
                        'bands': ['ndwi', 'ndwi', 'B4'],
                        'min': [-0.1, -0.1, 0],
                        'max': [0.5, 0.5, 3000],
                        'gamma': 1.6,
                        'dimensions': 3072,
                        'region': region,
                        'format': 'png'
                    })
                    
                    # False color to highlight vegetation, water, and bare earth
                    false_url = composite.getThumbURL({
                        'bands': ['B8', 'B11', 'B4'],
                        'min': 0,
                        'max': 3000,
                        'gamma': 1.5,
                        'dimensions': 3072,
                        'region': region,
                        'format': 'png'
                    })
                    
                    # Download and enhance images
                    try:
                        # Download true color image
                        response = requests.get(rgb_url)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            
                            # Apply advanced image enhancements
                            img = self.enhance_image_quality(img)
                            
                            # Save the high-quality image
                            filename = f'river_segment_{i+1}_true_color.png'
                            img.save(os.path.join(self.river_images_folder, filename), 
                                    format='PNG', optimize=True, quality=95)
                            
                            print(f"  Saved true color image: {filename}")
                        
                        # Download NDWI visualization
                        response = requests.get(ndwi_url)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            
                            # Apply basic enhancements
                            enhancer = ImageEnhance.Contrast(img)
                            img = enhancer.enhance(1.3)
                            
                            filename = f'river_segment_{i+1}_ndwi.png'
                            img.save(os.path.join(self.river_images_folder, filename), 
                                    format='PNG', optimize=True, quality=95)
                            
                            print(f"  Saved NDWI visualization: {filename}")
                        
                        # Download false color image
                        response = requests.get(false_url)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            
                            # Apply basic enhancements
                            enhancer = ImageEnhance.Contrast(img)
                            img = enhancer.enhance(1.2)
                            
                            filename = f'river_segment_{i+1}_false_color.png'
                            img.save(os.path.join(self.river_images_folder, filename), 
                                    format='PNG', optimize=True, quality=95)
                            
                            print(f"  Saved false color image: {filename}")
                            
                    except Exception as img_err:
                        print(f"  Error saving images: {str(img_err)}")
                else:
                    print(f"  No suitable imagery found for segment {i+1}")
                
                # Sleep to avoid rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing segment {i+1}: {str(e)}")
                continue
        
        print(f"Downloaded river-focused images to: {self.river_images_folder}")
        return True
    
    def enhance_image_quality(self, img):
        """Apply advanced image enhancements to improve visibility and reduce graininess"""
        try:
            # First, apply a very slight blur to reduce noise
            img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
            
            # Apply unsharp mask for more detail
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.4)
            
            # Boost color saturation to make features stand out
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.3)
            
            # Adjust brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
            
            # Final sharpening
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
            
            return img
        except Exception as e:
            print(f"Error during image enhancement: {str(e)}")
            return img  # Return original image if enhancement fails
    
    def create_labeling_interface(self):
        """Create a simple interface to view and label the river images"""
        class RiverImageViewer:
            def __init__(self, images_folder):
                self.images_folder = images_folder
                
                # Get all images
                self.true_color_images = sorted([f for f in os.listdir(images_folder) 
                                              if f.endswith('.png') and 'true_color' in f])
                self.ndwi_images = sorted([f for f in os.listdir(images_folder) 
                                        if f.endswith('.png') and 'ndwi' in f])
                self.false_color_images = sorted([f for f in os.listdir(images_folder) 
                                               if f.endswith('.png') and 'false_color' in f])
                
                self.current_index = 0
                self.current_view = "true_color"  # Default view
                
                # Create main window
                self.root = tk.Tk()
                self.root.title("River Sand Mining Image Viewer")
                
                # Make window full screen
                self.root.attributes('-fullscreen', True)
                
                # Exit fullscreen with ESC key
                self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
                
                # Main container
                self.main_frame = tk.Frame(self.root)
                self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
                
                # Progress and info
                info_frame = tk.Frame(self.main_frame)
                info_frame.pack(fill="x", pady=10)
                
                self.progress_label = tk.Label(info_frame, 
                                             text="", 
                                             font=("Arial", 16, "bold"))
                self.progress_label.pack(side=tk.LEFT, padx=20)
                
                # View toggle buttons
                view_frame = tk.Frame(info_frame)
                view_frame.pack(side=tk.RIGHT, padx=20)
                
                self.true_color_btn = tk.Button(view_frame, text="True Color", 
                                              command=lambda: self.switch_view("true_color"),
                                              bg="lightblue", width=15)
                self.true_color_btn.pack(side=tk.LEFT, padx=5)
                
                self.ndwi_btn = tk.Button(view_frame, text="Water Index", 
                                        command=lambda: self.switch_view("ndwi"),
                                        width=15)
                self.ndwi_btn.pack(side=tk.LEFT, padx=5)
                
                self.false_color_btn = tk.Button(view_frame, text="False Color", 
                                               command=lambda: self.switch_view("false_color"),
                                               width=15)
                self.false_color_btn.pack(side=tk.LEFT, padx=5)
                
                # Canvas with scrollbars for the image
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
                
                # Navigation and control buttons
                nav_frame = tk.Frame(self.main_frame)
                nav_frame.pack(pady=10)
                
                self.prev_btn = tk.Button(nav_frame, text="← Previous", command=self.prev_image,
                                       font=("Arial", 12), width=15, height=2)
                self.prev_btn.pack(side=tk.LEFT, padx=20)
                
                self.next_btn = tk.Button(nav_frame, text="Next →", command=self.next_image,
                                       font=("Arial", 12), width=15, height=2)
                self.next_btn.pack(side=tk.LEFT, padx=20)
                
                # Exit button
                exit_btn = tk.Button(nav_frame, text="Exit", command=self.root.destroy,
                                   font=("Arial", 12), width=15, height=2, bg="lightgray")
                exit_btn.pack(side=tk.LEFT, padx=20)
                
                # Load first image
                self.root.after(100, self.load_current_image)
                
                # Start event loop
                self.root.mainloop()
            
            def switch_view(self, view_type):
                """Switch between true color, NDWI, and false color views"""
                self.current_view = view_type
                
                # Update button colors
                self.true_color_btn.config(bg="white")
                self.ndwi_btn.config(bg="white")
                self.false_color_btn.config(bg="white")
                
                if view_type == "true_color":
                    self.true_color_btn.config(bg="lightblue")
                elif view_type == "ndwi":
                    self.ndwi_btn.config(bg="lightblue")
                elif view_type == "false_color":
                    self.false_color_btn.config(bg="lightblue")
                
                self.load_current_image()
            
            def load_current_image(self):
                """Load the current image based on index and view type"""
                if self.current_view == "true_color" and len(self.true_color_images) > 0:
                    images = self.true_color_images
                elif self.current_view == "ndwi" and len(self.ndwi_images) > 0:
                    images = self.ndwi_images
                elif self.current_view == "false_color" and len(self.false_color_images) > 0:
                    images = self.false_color_images
                else:
                    # Fallback to true color if selected view isn't available
                    images = self.true_color_images
                    self.current_view = "true_color"
                    self.true_color_btn.config(bg="lightblue")
                
                if not images:
                    self.progress_label.config(text="No images available")
                    return
                
                # Keep index within bounds
                if self.current_index >= len(images):
                    self.current_index = len(images) - 1
                elif self.current_index < 0:
                    self.current_index = 0
                
                # Update progress indicator
                progress_text = f"Image {self.current_index + 1} of {len(images)} | View: {self.current_view.replace('_', ' ').title()}"
                self.progress_label.config(text=progress_text)
                
                # Load and display image
                img_path = os.path.join(self.images_folder, images[self.current_index])
                image = Image.open(img_path)
                
                # Convert to PhotoImage
                self.photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            
            def next_image(self):
                """Show next image"""
                if self.current_view == "true_color":
                    max_index = len(self.true_color_images) - 1
                elif self.current_view == "ndwi":
                    max_index = len(self.ndwi_images) - 1
                elif self.current_view == "false_color":
                    max_index = len(self.false_color_images) - 1
                else:
                    max_index = 0
                
                if self.current_index < max_index:
                    self.current_index += 1
                    self.load_current_image()
            
            def prev_image(self):
                """Show previous image"""
                if self.current_index > 0:
                    self.current_index -= 1
                    self.load_current_image()
        
        # Start the viewer
        RiverImageViewer(self.river_images_folder)
    
def main():
    try:
        print("Starting River-Focused Sand Mining Detection...")
        
        river_detector = RiverFocusedSandMining()
        
        # Download high-quality river images
        shapefile_path = input("Enter path to river shapefile: ")
        
        if not os.path.exists(shapefile_path):
            print(f"Error: File not found at {shapefile_path}")
            return
        
        distance_km = float(input("Enter sampling distance in km (default 5): ") or "5")
        
        print(f"\nDownloading river-focused images every {distance_km}km...")
        success = river_detector.download_river_focused_images(shapefile_path, distance_km)
        
        if success:
            print("\nWould you like to view the downloaded images? (y/n)")
            view_images = input().strip().lower()
            
            if view_images == 'y' or view_images == 'yes':
                river_detector.create_labeling_interface()
        
        print("Done!")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have:")
        print("1. Signed up for Google Earth Engine")
        print("2. Created a Google Cloud Project")
        print("3. Enabled the Earth Engine API")
        print("4. Authenticated properly")

if __name__ == "__main__":
    main()