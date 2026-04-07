"""
Data Fusion Script for Rockfall Prediction System
Merges slope stability, weather, and image datasets into master training dataset
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List

def create_sample_datasets():
    """Create sample datasets for demonstration since actual datasets aren't provided"""
    
    # Create sample slope stability dataset
    np.random.seed(42)
    n_samples = 500
    
    slope_data = {
        'slope_angle': np.random.uniform(15, 45, n_samples),
        'pore_water_pressure': np.random.uniform(0.1, 0.9, n_samples),
        'factor_of_safety': np.random.uniform(0.5, 2.0, n_samples),
        'location_id': [f'slope_{i}' for i in range(n_samples)]
    }
    slope_df = pd.DataFrame(slope_data)
    slope_df.to_csv('slope_data.csv', index=False)
    
    # Create sample weather dataset (NASA-style)
    weather_data = {
        'prectot': np.random.uniform(0, 100, n_samples),  # precipitation
        'temperature': np.random.uniform(10, 35, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'soil_wetness': np.random.uniform(0.1, 0.8, n_samples),
        'location_id': [f'weather_{i}' for i in range(n_samples)]
    }
    weather_df = pd.DataFrame(weather_data)
    weather_df.to_csv('nasa_weather.csv', index=False)
    
    # Create sample image annotations
    image_annotations = []
    for i in range(n_samples):
        num_boxes = np.random.randint(0, 5)
        boxes = []
        for j in range(num_boxes):
            boxes.append({
                'label': int(np.random.choice([0, 1])),  # 0=stable, 1=unstable
                'bbox': [int(np.random.randint(0, 100)), int(np.random.randint(0, 100)), 
                        int(np.random.randint(20, 50)), int(np.random.randint(20, 50))]
            })
        
        image_annotations.append({
            'image_id': f'img_{i}',
            'annotations': boxes
        })
    
    with open('image_annotations.json', 'w') as f:
        json.dump(image_annotations, f, indent=2)
    
    return slope_df, weather_df, image_annotations

def extract_instability_scores(annotation_file: str) -> pd.DataFrame:
    """Extract instability scores from image annotation JSON"""
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    instability_data = []
    for img_data in annotations:
        img_id = img_data['image_id']
        unstable_count = sum(1 for ann in img_data['annotations'] if ann['label'] == 1)
        instability_data.append({
            'image_id': img_id,
            'instability_score': unstable_count
        })
    
    return pd.DataFrame(instability_data)

def create_master_dataset(n_scenarios: int = 1000) -> pd.DataFrame:
    """Create master training dataset by sampling from all three sources"""
    
    # Load datasets
    try:
        slope_df = pd.read_csv('slope_data.csv')
        weather_df = pd.read_csv('nasa_weather.csv')
        image_features = extract_instability_scores('image_annotations.json')
    except FileNotFoundError:
        print("Sample datasets not found. Creating sample data...")
        slope_df, weather_df, _ = create_sample_datasets()
        image_features = extract_instability_scores('image_annotations.json')
    
    # Create master dataframe with randomized alignment
    master_df = pd.DataFrame()
    
    # Sample from each dataset to create mining scenarios
    master_df['slope_angle'] = slope_df['slope_angle'].sample(n_scenarios, replace=True, random_state=42).values
    master_df['pore_pressure'] = slope_df['pore_water_pressure'].sample(n_scenarios, replace=True, random_state=43).values
    master_df['factor_of_safety'] = slope_df['factor_of_safety'].sample(n_scenarios, replace=True, random_state=44).values
    
    master_df['rainfall'] = weather_df['prectot'].sample(n_scenarios, replace=True, random_state=45).values
    master_df['temperature'] = weather_df['temperature'].sample(n_scenarios, replace=True, random_state=46).values
    master_df['humidity'] = weather_df['humidity'].sample(n_scenarios, replace=True, random_state=47).values
    master_df['soil_wetness'] = weather_df['soil_wetness'].sample(n_scenarios, replace=True, random_state=48).values
    
    master_df['instability_score'] = image_features['instability_score'].sample(n_scenarios, replace=True, random_state=49).values
    
    # Create risk labels using logic-based approach
    # High risk if: factor_of_safety < 1 OR (high pore_pressure AND high rainfall) OR instability_score > 2
    master_df['risk_label'] = np.where(
        (master_df['factor_of_safety'] < 1.0) |
        ((master_df['pore_pressure'] > 0.7) & (master_df['rainfall'] > 50)) |
        (master_df['instability_score'] > 2),
        1, 0
    )
    
    # Add scenario IDs for tracking
    master_df['scenario_id'] = [f'scenario_{i}' for i in range(n_scenarios)]
    
    return master_df

def main():
    """Main function to create and save the master dataset"""
    
    print("Creating master training dataset...")
    
    # Create master dataset
    master_df = create_master_dataset(n_scenarios=1000)
    
    # Save to CSV
    master_df.to_csv('master_train.csv', index=False)
    
    # Display dataset statistics
    print(f"\nDataset created with {len(master_df)} scenarios")
    print(f"Risk distribution: {master_df['risk_label'].value_counts().to_dict()}")
    print(f"Risk percentage: {master_df['risk_label'].mean() * 100:.1f}%")
    
    # Display sample data
    print("\nSample data:")
    print(master_df.head())
    
    # Display feature statistics
    print("\nFeature statistics:")
    print(master_df.describe())
    
    print("\nMaster dataset saved as 'master_train.csv'")
    
    return master_df

if __name__ == "__main__":
    master_df = main()
