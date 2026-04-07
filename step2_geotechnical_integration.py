"""
Step 2: Integrate Geotechnical Data with Weather Data
Downloads Kaggle slope stability dataset and fuses with weather using physics-based logic
"""

import pandas as pd
import numpy as np
import os
import kagglehub

def download_geotechnical_data():
    """Download slope stability dataset from Kaggle"""
    
    print("📥 Downloading geotechnical data from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("ziya07/slope-stability-analysis-dataset")
    print(f"Path to dataset files: {path}")
    
    # Find CSV file
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset")
    
    slope_csv_path = os.path.join(path, csv_files[0])
    geotech_df = pd.read_csv(slope_csv_path)
    
    print(f"✅ Geotechnical data loaded: {len(geotech_df)} rows")
    print(f"📋 Columns found: {geotech_df.columns.tolist()}")
    
    return geotech_df, slope_csv_path

def load_weather_data():
    """Load existing weather data"""
    
    try:
        weather_df = pd.read_csv('data/india_mining_regions_weather_combined.csv')
        print(f"✅ Weather data loaded: {len(weather_df)} rows")
        return weather_df
    except FileNotFoundError:
        print("❌ Weather data not found. Please run data_fusion.py first.")
        return None

def apply_physics_based_fusion(weather_df, geotech_df):
    """Apply physics-based logic to fuse weather and geotechnical data"""
    
    print("🔬 Applying physics-based fusion logic...")
    
    # Create fused dataset starting with weather data
    fused_df = weather_df.copy()
    
    # Deterministic mapping (non-random): Rank-order / CDF alignment
    # High rainfall days are matched with high pore-pressure states.
    n_weather_rows = len(fused_df)
    
    # Ensure correct types + sort weather by rainfall severity
    fused_df['rainfall_mm'] = pd.to_numeric(fused_df['rainfall_mm'], errors='coerce').fillna(0.0)
    fused_df = fused_df.sort_values(['rainfall_mm', 'date', 'region']).reset_index(drop=True)
    
    # Sort geotech by pore pressure severity (low -> high)
    geotech_sorted = geotech_df.copy()
    geotech_sorted['Pore Water Pressure Ratio'] = pd.to_numeric(
        geotech_sorted['Pore Water Pressure Ratio'], errors='coerce'
    )
    geotech_sorted = geotech_sorted.dropna(subset=['Pore Water Pressure Ratio']).sort_values(
        ['Pore Water Pressure Ratio']
    ).reset_index(drop=True)
    
    # Map each weather row to a geotech row by percentile rank
    geotech_indices = np.linspace(0, len(geotech_sorted) - 1, n_weather_rows)
    geotech_indices = np.round(geotech_indices).astype(int)
    
    fused_df['pore_pressure_ratio'] = geotech_sorted.iloc[geotech_indices]['Pore Water Pressure Ratio'].values
    fused_df['slope_angle'] = geotech_sorted.iloc[geotech_indices]['Slope Angle (°)'].values
    fused_df['slope_height_m'] = geotech_sorted.iloc[geotech_indices]['Slope Height (m)'].values
    fused_df['cohesion_kpa'] = geotech_sorted.iloc[geotech_indices]['Cohesion (kPa)'].values
    fused_df['internal_friction_angle_deg'] = geotech_sorted.iloc[geotech_indices]['Internal Friction Angle (°)'].values
    fused_df['unit_weight_kn_m3'] = geotech_sorted.iloc[geotech_indices]['Unit Weight (kN/m³)'].values
    fused_df['reinforcement_type'] = geotech_sorted.iloc[geotech_indices]['Reinforcement Type'].values
    fused_df['reinforcement_numeric'] = geotech_sorted.iloc[geotech_indices]['Reinforcement Numeric'].values
    fused_df['base_fs'] = geotech_sorted.iloc[geotech_indices]['Factor of Safety (FS)'].values
    
    # Antecedent rainfall features (lag effect)
    # Rolling rainfall computed per-region to preserve local rainfall dynamics.
    fused_df['date'] = pd.to_datetime(fused_df['date'], errors='coerce')
    fused_df['rainfall_1d'] = fused_df['rainfall_mm']
    fused_df['rainfall_3d_avg'] = (
        fused_df.groupby('region', sort=False)['rainfall_mm']
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    
    # Physics Logic: rainfall increases pore pressure
    # Use 3-day average to represent infiltration lag.
    fused_df['adjusted_pore_pressure'] = fused_df['pore_pressure_ratio'] + (fused_df['rainfall_mm'] * 0.005)
    fused_df['adjusted_pore_pressure'] = fused_df['adjusted_pore_pressure'].clip(0, 1)  # Keep between 0-1

    # Dynamic risk label (mentor-approved): keep Kaggle FS as a feature, but define risk using FS + rainfall triggers
    # This keeps validated geotech physics (FS) while still letting environment (rain) drive the hazard label.
    def calculate_risk(row):
        fs = row['base_fs']
        rain = row['rainfall_mm']
        rain_3d = row['rainfall_3d_avg']

        # Already failing
        if fs < 1.0:
            return 1

        # Near-critical slope + heavy rain trigger (use either same-day or antecedent rain)
        if fs < 1.3 and (rain > 50 or rain_3d > 30):
            return 1

        return 0

    fused_df['risk_label'] = fused_df.apply(calculate_risk, axis=1)
    
    # Add scenario IDs
    fused_df['scenario_id'] = [f'scenario_{i}' for i in range(len(fused_df))]
    
    print(f"✅ Physics-based fusion complete: {len(fused_df)} scenarios")
    print(f"📊 Risk distribution: {fused_df['risk_label'].value_counts().to_dict()}")
    print(f"🎯 Risk percentage: {fused_df['risk_label'].mean() * 100:.1f}%")
    
    return fused_df

def analyze_fused_dataset(df):
    """Analyze the fused dataset"""
    
    print(f"\n📈 Fused Dataset Analysis")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total scenarios: {len(df):,}")
    print(f"Risk distribution: {df['risk_label'].value_counts().to_dict()}")
    print(f"Risk percentage: {df['risk_label'].mean() * 100:.1f}%")
    
    # Regional analysis
    print(f"\n🌍 Regional Analysis:")
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        print(f"\n   {region}:")
        print(f"     Scenarios: {len(region_data):,}")
        print(f"     Risk %: {region_data['risk_label'].mean() * 100:.1f}%")
        print(f"     Avg Rainfall: {region_data['rainfall_mm'].mean():.1f} mm")
        print(f"     Avg Pore Pressure: {region_data['adjusted_pore_pressure'].mean():.3f}")
        print(f"     Avg Base FS: {region_data['base_fs'].mean():.2f}")
        print(f"     Avg Slope Angle: {region_data['slope_angle'].mean():.1f}°")
    
    # Correlation analysis
    print(f"\n🔗 Correlation with Risk:")
    numeric_cols = ['rainfall_mm', 'temperature_c', 'humidity_percent', 'adjusted_pore_pressure',
                   'slope_angle', 'cohesion_kpa', 'base_fs']
    correlations = df[numeric_cols + ['risk_label']].corr()['risk_label'].sort_values(ascending=False)
    
    for col, corr in correlations.items():
        if col != 'risk_label':
            print(f"   {col}: {corr:.3f}")

def main():
    """Main function for Step 2"""
    
    print("🏗️ Step 2: Integrate Geotechnical Data with Weather")
    print("Applying physics-based pore pressure logic")
    print("=" * 60)
    
    # Step 1: Download geotechnical data
    try:
        geotech_df, csv_path = download_geotechnical_data()
    except Exception as e:
        print(f"❌ Failed to download geotechnical data: {e}")
        return None
    
    # Step 2: Load weather data
    weather_df = load_weather_data()
    if weather_df is None:
        return None
    
    # Step 3: Apply physics-based fusion
    fused_df = apply_physics_based_fusion(weather_df, geotech_df)
    
    # Step 4: Analyze results
    analyze_fused_dataset(fused_df)
    
    # Step 5: Save results
    output_file = 'data/weather_geotech_combined.csv'
    fused_df.to_csv(output_file, index=False)
    print(f"\n💾 Fused dataset saved as: {output_file}")
    
    # Display sample data
    print(f"\n📋 Sample Fused Data:")
    display_cols = ['scenario_id', 'date', 'region', 'rainfall_mm', 'rainfall_3d_avg',
                   'adjusted_pore_pressure', 'slope_angle', 'base_fs', 'risk_label']
    print(fused_df[display_cols].head(10).to_string(index=False))
    
    print(f"\n✅ Step 2 completed successfully!")
    print(f"📁 Generated file: {output_file}")
    print(f"📝 Ready for Step 3: Add imagery instability scores")
    
    return fused_df

if __name__ == "__main__":
    fused_df = main()
