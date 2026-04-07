"""
Enhanced Data Fusion Script for Rockfall Prediction System
Combines weather data loading and fusion into single process
Uses real weather data from 4 Indian mining regions
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List

def read_nasa_weather_csv(filepath, region_name):
    """Read NASA POWER weather CSV file and extract location info"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract location information from header
    location_info = {}
    for line in lines:
        if 'Location:' in line:
            # Extract coordinates - handle the specific format
            # Line format: Location: latitude  23.7957   longitude 86.4304
            parts = line.split()
            try:
                lat_idx = parts.index('latitude')
                lon_idx = parts.index('longitude')
                location_info['latitude'] = float(parts[lat_idx + 1])
                location_info['longitude'] = float(parts[lon_idx + 1])
            except (ValueError, IndexError):
                # Fallback parsing
                import re
                coords = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if len(coords) >= 2:
                    location_info['latitude'] = float(coords[0])
                    location_info['longitude'] = float(coords[1])
        elif 'elevation' in line and '=' in line:
            # Extract elevation
            elev_str = line.split('=')[1].split('meters')[0].strip()
            location_info['elevation'] = float(elev_str)
    
    # Find the start of data (after -END HEADER-)
    data_start = 0
    for i, line in enumerate(lines):
        if '-END HEADER-' in line:
            data_start = i + 1
            break
    
    # Read the data portion
    data_lines = lines[data_start:]
    if not data_lines:
        return None
    
    # Parse CSV data
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(data_lines)))
    
    # Add region and location information
    df['region'] = region_name
    df['latitude'] = location_info.get('latitude', np.nan)
    df['longitude'] = location_info.get('longitude', np.nan)
    df['elevation'] = location_info.get('elevation', np.nan)
    
    return df

def convert_units_and_clean(df):
    """Convert units and clean the weather data"""
    
    # Convert precipitation from inches/day to mm/day (1 inch = 25.4 mm)
    df['rainfall_mm'] = df['PRECTOTCORR'] * 25.4
    
    # Convert temperature from Fahrenheit to Celsius
    df['temperature_c'] = (df['T2M'] - 32) * 5/9
    
    # Convert wind speed from mph to km/h (1 mph = 1.60934 km/h)
    df['wind_speed_kmh'] = df['WS2M'] * 1.60934
    
    # Rename columns for clarity
    df = df.rename(columns={
        'RH2M': 'humidity_percent',
        'GWETTOP': 'soil_wetness',
        'YEAR': 'year',
        'DOY': 'day_of_year'
    })
    
    # Replace -999 (missing data) with NaN
    df = df.replace(-999, np.nan)
    
    # Create date column from year and day of year
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['day_of_year'].astype(str), format='%Y-%j')
    
    # Select and reorder relevant columns
    final_columns = [
        'date', 'year', 'day_of_year', 'region', 'latitude', 'longitude', 'elevation',
        'rainfall_mm', 'temperature_c', 'humidity_percent', 'wind_speed_kmh', 'soil_wetness'
    ]
    
    df = df[final_columns]
    
    return df

def load_and_combine_weather_data():
    """Load and combine weather data from 4 Indian mining regions"""
    
    # File paths
    weather_files = {
        'Odisha': r"C:\Users\akshi\Downloads\Odisha (Keonjhar Iron Ore Mines).csv",
        'Jharkhand': r"C:\Users\akshi\Downloads\Jharkhand (Jharia Coalfield).csv",
        'Karnataka': r"C:\Users\akshi\Downloads\Karnataka (Bellary Iron Ore Belt).csv",
        'Chhattisgarh': r"C:\Users\akshi\Downloads\Chhattisgarh (Korba Coal Mines).csv"
    }
    
    all_data = []
    
    print("🌤️  Loading Weather Data from Indian Mining Regions")
    print("=" * 60)
    
    for region, filepath in weather_files.items():
        print(f"\n📍 Processing {region}...")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
        
        try:
            # Read and process the data
            df = read_nasa_weather_csv(filepath, region)
            
            if df is not None:
                # Convert units and clean
                df_clean = convert_units_and_clean(df)
                
                print(f"✅ {region}: {len(df_clean)} records loaded")
                print(f"   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
                print(f"   Location: {df_clean['latitude'].iloc[0]:.4f}°N, {df_clean['longitude'].iloc[0]:.4f}°E")
                
                all_data.append(df_clean)
            else:
                print(f"❌ Failed to read {region} data")
                
        except Exception as e:
            print(f"❌ Error processing {region}: {str(e)}")
    
    if not all_data:
        print("\n❌ No weather data was successfully loaded!")
        return None
    
    # Combine all dataframes
    print(f"\n🔗 Combining weather data from {len(all_data)} regions...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and region
    combined_df = combined_df.sort_values(['date', 'region']).reset_index(drop=True)
    
    # Save combined dataset
    output_file = 'data/india_mining_regions_weather_combined.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"💾 Combined weather data saved as: {output_file}")
    
    return combined_df

if __name__ == "__main__":
    weather_df = main()
