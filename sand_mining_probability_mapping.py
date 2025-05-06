"""
River Sand Mining Probability Mapper
===================================
This script creates a comprehensive probability map of sand mining activities along an entire river.
It loads a trained sand mining detection model and applies it to regularly spaced points along 
the entire river shapefile, using the latest available satellite imagery.

Usage:
    python sand_mining_probability_mapping.py --shapefile path/to/river.shp --output output_map.html

Requirements:
    - Pre-trained sand mining model
    - Google Earth Engine account
    - Required Python packages
"""

import ee
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import requests
import io
import time
import argparse
from datetime import datetime, timedelta
import joblib
import geopandas as gpd
from tqdm import tqdm
import folium
from folium.plugins import HeatMap, MarkerCluster
import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import json
import shutil
import warnings
warnings.filterwarnings('ignore')

class SandMiningProbabilityMapper:
    def __init__(self, model_path=None, base_folder='sand_mining_detection'):
        # Initialize Earth Engine with proper authentication
        try:
            ee.Initialize()
            print("Successfully initialized Earth Engine")
        except Exception as e:
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
            except Exception as auth_e:
                print(f"\nAuthentication failed: {str(auth_e)}")
                print("\nPlease follow these steps:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project or select an existing one")
                print("3. Enable the Earth Engine API for your project")
                print("4. Run this script again")
                raise auth_e
        
        # Set up folders
        self.base_folder = base_folder
        self.output_folder = os.path.join(self.base_folder, 'probability_maps')
        self.temp_folder = os.path.join(self.base_folder, 'temp')
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        
        # Load model
        if model_path is None:
            model_path = os.path.join(self.base_folder, 'sand_mining_model.joblib')
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first using the main sand mining detection script.")
            return
        
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    def get_latest_imagery_date(self, region):
        """Determine the latest available imagery date with acceptable cloud coverage"""
        try:
            # First try Sentinel-2 Harmonized
            today = datetime.now()
            
            # Start with last 3 months
            start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            # Try to find recent images
            collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterDate(start_date, end_date) \
                .filterBounds(region) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            
            count = collection.size().getInfo()
            
            # If no recent images, try last year
            if count == 0:
                start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d') 
                collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(region) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                
                count = collection.size().getInfo()
            
            # If still no images, try regular Sentinel-2
            if count == 0:
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(region) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                
                count = collection.size().getInfo()
            
            # Get the most recent image date
            if count > 0:
                latest_image = collection.sort('system:time_start', False).first()
                date_millis = latest_image.get('system:time_start').getInfo()
                latest_date = datetime.fromtimestamp(date_millis/1000).strftime('%Y-%m-%d')
                source = "Sentinel-2"
                return latest_date, source
            else:
                # Fall back to Landsat if needed
                collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                    .filterDate(start_date, end_date) \
                    .filterBounds(region)
                
                if collection.size().getInfo() > 0:
                    latest_image = collection.sort('system:time_start', False).first()
                    date_millis = latest_image.get('system:time_start').getInfo()
                    latest_date = datetime.fromtimestamp(date_millis/1000).strftime('%Y-%m-%d')
                    source = "Landsat 8"
                    return latest_date, source
                else:
                    # Just return today's date as a fallback
                    return today.strftime('%Y-%m-%d'), "No recent imagery"
                
        except Exception as e:
            print(f"Error determining latest imagery date: {e}")
            # Return today's date as a fallback
            return datetime.now().strftime('%Y-%m-%d'), "Error"
    
    def generate_river_points(self, shapefile_path, distance_km=1):
        """Generate dense sampling points along the entire river"""
        try:
            print(f"Loading shapefile from: {shapefile_path}")
            gdf = gpd.read_file(shapefile_path)
            
            # Get the river geometry
            if isinstance(gdf.geometry.iloc[0], gpd.geoseries.GeoSeries):
                river_geom = gdf.geometry.iloc[0]
            else:
                river_geom = gdf.geometry.union_all()
            
            coordinates = []
            
            # If the geometry is a MultiLineString or MultiPolygon, convert to single geometry
            if river_geom.geom_type == 'MultiLineString':
                # Process each line segment
                lines = list(river_geom.geoms)
                for line in lines:
                    coords = self._sample_line(line, distance_km)
                    coordinates.extend(coords)
            elif river_geom.geom_type == 'MultiPolygon':
                # Convert MultiPolygon to single Polygon by taking the largest
                polygons = list(river_geom.geoms)
                river_geom = max(polygons, key=lambda x: x.area)
                coords = self._sample_polygon(river_geom, distance_km)
                coordinates.extend(coords)
            elif river_geom.geom_type == 'LineString':
                coords = self._sample_line(river_geom, distance_km)
                coordinates.extend(coords)
            elif river_geom.geom_type == 'Polygon':
                coords = self._sample_polygon(river_geom, distance_km)
                coordinates.extend(coords)
            else:
                print(f"Unexpected geometry type: {river_geom.geom_type}")
                return []
            
            print(f"Generated {len(coordinates)} sampling points along the river")
            
            # Create GeoDataFrame for these points
            points_gdf = gpd.GeoDataFrame(
                geometry=[gpd.points.Point(lon, lat) for lat, lon in coordinates],
                crs=gdf.crs
            )
            points_gdf['latitude'] = [lat for lat, lon in coordinates]
            points_gdf['longitude'] = [lon for lat, lon in coordinates]
            
            # Save points to shapefile for reference
            points_shapefile = os.path.join(self.output_folder, 'river_sampling_points.shp')
            points_gdf.to_file(points_shapefile)
            print(f"Sampling points saved to: {points_shapefile}")
            
            return coordinates
            
        except Exception as e:
            print(f"Error generating river points: {e}")
            return []
    
    def _sample_line(self, line, distance_km):
        """Sample points along a LineString at regular intervals"""
        coords = list(line.coords)
        
        # Convert distance to degrees (approximation: 1 degree ≈ 111 km)
        distance_degrees = distance_km / 111.0
        
        # Calculate total length
        total_length = 0
        for i in range(len(coords) - 1):
            segment_length = ((coords[i+1][0] - coords[i][0])**2 + 
                             (coords[i+1][1] - coords[i][1])**2)**0.5
            total_length += segment_length
        
        # Calculate points along the line
        points = []
        current_distance = 0
        target_distance = 0
        
        for i in range(len(coords) - 1):
            start_coord = coords[i]
            end_coord = coords[i + 1]
            
            segment_length = ((end_coord[0] - start_coord[0])**2 + 
                             (end_coord[1] - start_coord[1])**2)**0.5
            
            while current_distance + segment_length >= target_distance:
                # Interpolate point along the segment
                t = (target_distance - current_distance) / segment_length
                lon = start_coord[0] + t * (end_coord[0] - start_coord[0])
                lat = start_coord[1] + t * (end_coord[1] - start_coord[1])
                
                points.append((lat, lon))
                target_distance += distance_degrees
                
                if target_distance > total_length:
                    break
            
            current_distance += segment_length
            if target_distance > total_length:
                break
        
        return points
    
    def _sample_polygon(self, polygon, distance_km):
        """Sample points along a Polygon's exterior at regular intervals"""
        return self._sample_line(polygon.exterior, distance_km)
    
    def analyze_point(self, lat, lon, latest_date):
        """Download image and analyze a specific point"""
        try:
            # Create Earth Engine point and region
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(1000)  # 1000m buffer = 2km area
            
            # Get imagery from the latest date
            start_date = datetime.strptime(latest_date, '%Y-%m-%d') - timedelta(days=30)
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # Try Sentinel-2 Harmonized first
            s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
                .filterDate(start_date_str, latest_date) \
                .filterBounds(region) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE') \
                .first()
            
            # Fall back to regular Sentinel-2 if needed
            if s2 is None:
                s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date_str, latest_date) \
                    .filterBounds(region) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE') \
                    .first()
            
            # If still no image, try with longer time range
            if s2 is None:
                start_date = datetime.strptime(latest_date, '%Y-%m-%d') - timedelta(days=180)
                start_date_str = start_date.strftime('%Y-%m-%d')
                
                s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date_str, latest_date) \
                    .filterBounds(region) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE') \
                    .first()
            
            # Download image
            if s2 is not None:
                rgb_url = s2.getThumbURL({
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.4,
                    'bands': ['B4', 'B3', 'B2'],  # True color
                    'dimensions': 800,  # Smaller for faster processing
                    'region': region,
                    'format': 'png'
                })
                
                response = requests.get(rgb_url)
                img = Image.open(io.BytesIO(response.content))
                
                # Enhance image
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.4)
                
                # Save temporarily
                temp_file = os.path.join(self.temp_folder, f'temp_{lat}_{lon}.png')
                img.save(temp_file)
                
                # Extract features and predict
                from improved_sand_mining_detection import SandMiningDetection
                detector = SandMiningDetection()
                features = detector.extract_enhanced_features(temp_file)
                
                # Make prediction
                prediction = self.model.predict([features])[0]
                probability = self.model.predict_proba([features])[0][1]  # Probability of sand mining
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                return probability
            else:
                print(f"Warning: No suitable imagery found for point {lat}, {lon}")
                return None
            
        except Exception as e:
            print(f"Error analyzing point {lat}, {lon}: {e}")
            return None
    
    def create_probability_map(self, shapefile_path, distance_km=1, output_file=None):
        """Generate a comprehensive probability map of sand mining along the river"""
        print("Starting sand mining probability mapping...")
        
        # Get the latest imagery date
        # Use the river's bounding box to search for imagery
        gdf = gpd.read_file(shapefile_path)
        bounds = gdf.total_bounds
        region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
        latest_date, source = self.get_latest_imagery_date(region)
        
        print(f"Using latest available imagery from: {latest_date} (Source: {source})")
        
        # Generate sampling points along the river
        coords = self.generate_river_points(shapefile_path, distance_km)
        
        if not coords:
            print("Error: Failed to generate river points")
            return None
        
        # Create a DataFrame to store results
        results = []
        
        # Process each point
        print(f"Analyzing {len(coords)} points along the river...")
        for i, (lat, lon) in enumerate(tqdm(coords)):
            # Analyze point (with a small delay to respect rate limits)
            probability = self.analyze_point(lat, lon, latest_date)
            
            if probability is not None:
                results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'probability': probability,
                    'classification': 'Sand Mining' if probability > 0.5 else 'No Sand Mining'
                })
                
            # Add delay to avoid rate limits
            if i % 10 == 0:
                time.sleep(1)
        
        # Create DataFrame and save to CSV
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.output_folder, f'sand_mining_probabilities_{latest_date}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved results to: {csv_path}")
            
            # Create interactive map
            if output_file is None:
                output_file = os.path.join(self.output_folder, f'sand_mining_map_{latest_date}.html')
            
            self.generate_interactive_map(df, shapefile_path, output_file, latest_date)
            print(f"Interactive map saved to: {output_file}")
            
            return df
        else:
            print("Error: No valid results generated")
            return None
    
    def generate_interactive_map(self, results_df, shapefile_path, output_file, imagery_date):
        """Create an interactive Folium map with the results"""
        # Load the river shapefile
        river_gdf = gpd.read_file(shapefile_path)
        
        # Calculate map center
        center_lat = results_df['latitude'].mean()
        center_lon = results_df['longitude'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                     tiles='CartoDB positron')
        
        # Create a colormap for probability values
        color_scale = cm.LinearColormap(
            ['green', 'yellow', 'orange', 'red'],
            vmin=0, vmax=1,
            caption='Sand Mining Probability'
        )
        m.add_child(color_scale)
        
        # Add river geometry
        folium.GeoJson(
            river_gdf,
            name='River',
            style_function=lambda x: {
                'color': 'blue',
                'weight': 3,
                'opacity': 0.7,
            }
        ).add_to(m)
        
        # Create marker cluster for easier navigation
        marker_cluster = MarkerCluster(name="Sampling Points").add_to(m)
        
        # Add points as markers with probability info
        for idx, row in results_df.iterrows():
            color = color_scale(row['probability'])
            
            # Create circle marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>Coordinates:</b> {row['latitude']:.5f}, {row['longitude']:.5f}<br>"
                    f"<b>Sand Mining Probability:</b> {row['probability']:.2f}<br>"
                    f"<b>Classification:</b> {row['classification']}<br>"
                    f"<b>Imagery Date:</b> {imagery_date}",
                    max_width=300
                )
            ).add_to(marker_cluster)
        
        # Add heatmap layer
        heatmap_data = results_df[['latitude', 'longitude', 'probability']].values.tolist()
        HeatMap(
            heatmap_data,
            name='Sand Mining Heatmap',
            radius=15,
            blur=10,
            gradient={0.0: 'green', 0.5: 'yellow', 0.7: 'orange', 1: 'red'}
        ).add_to(m)
        
        # Add probability threshold layers
        high_risk = results_df[results_df['probability'] >= 0.7]
        medium_risk = results_df[(results_df['probability'] >= 0.5) & (results_df['probability'] < 0.7)]
        
        if not high_risk.empty:
            folium.FeatureGroup(name="High Risk Areas (p ≥ 0.7)").add_to(m)
            for _, row in high_risk.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=12,
                    color='red',
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"High Risk: {row['probability']:.2f}"
                ).add_to(m)
        
        if not medium_risk.empty:
            folium.FeatureGroup(name="Medium Risk Areas (0.5 ≤ p < 0.7)").add_to(m)
            for _, row in medium_risk.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=10,
                    color='orange',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"Medium Risk: {row['probability']:.2f}"
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add title and info
        title_html = f'''
             <h3 align="center" style="font-size:20px"><b>Sand Mining Probability Map</b></h3>
             <h4 align="center" style="font-size:16px">Date: {imagery_date}</h4>
             <div align="center" style="font-size:14px">
                Total points analyzed: {len(results_df)}<br>
                High risk locations: {len(high_risk)} ({len(high_risk)/len(results_df)*100:.1f}%)<br>
                Medium risk locations: {len(medium_risk)} ({len(medium_risk)/len(results_df)*100:.1f}%)<br>
             </div>
             '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save the map
        m.save(output_file)
        return output_file
    
    def create_report(self, results_df, shapefile_path, imagery_date, output_file=None):
        """Generate a PDF report with analysis results"""
        if output_file is None:
            output_file = os.path.join(self.output_folder, f'sand_mining_report_{imagery_date}.pdf')
        
        # This is a placeholder - in a real implementation, you'd use 
        # libraries like reportlab, matplotlib, etc. to create a detailed PDF report
        print(f"Report generation would save to: {output_file}")
        
        # For now, we'll just generate a simple summary plot
        plt.figure(figsize=(12, 8))
        
        # Create histogram of probabilities
        plt.subplot(2, 2, 1)
        plt.hist(results_df['probability'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Sand Mining Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        
        # Create bar chart of classifications
        plt.subplot(2, 2, 2)
        classification_counts = results_df['classification'].value_counts()
        classification_counts.plot(kind='bar', color=['green', 'red'])
        plt.title('Classification Results')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.output_folder, f'sand_mining_summary_{imagery_date}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved to: {plot_file}")
        
        return plot_file

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate sand mining probability map for a river')
    parser.add_argument('--shapefile', type=str, required=True, help='Path to river shapefile')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model file')
    parser.add_argument('--distance', type=float, default=1.0, help='Sampling distance in km (default: 1km)')
    parser.add_argument('--output', type=str, default=None, help='Output map filename')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create mapper
    mapper = SandMiningProbabilityMapper(model_path=args.model)
    
    # Generate probability map
    results_df = mapper.create_probability_map(
        shapefile_path=args.shapefile,
        distance_km=args.distance,
        output_file=args.output
    )
    
    if results_df is not None:
        print("\nSummary of Sand Mining Detection:")
        print(f"Total points analyzed: {len(results_df)}")
        print(f"Points classified as sand mining: {sum(results_df['probability'] > 0.5)}")
        print(f"Points classified as non-sand mining: {sum(results_df['probability'] <= 0.5)}")
        print(f"Average probability: {results_df['probability'].mean():.3f}")
        
        # Generate a basic report with plots
        if args.output:
            report_file = mapper.create_report(results_df, args.shapefile, datetime.now().strftime('%Y-%m-%d'))
            print(f"\nReport generated: {report_file}")
            
        print("\nDone! You can view the interactive map in your browser.")
    else:
        print("Failed to generate probability map.")

if __name__ == "__main__":
    main()