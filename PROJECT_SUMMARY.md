# Rockfall Prediction System - Project Summary

## 🎯 Project Overview
This project implements a comprehensive rockfall prediction system for open-pit mines, integrating multi-source data (geotechnical, environmental, and visual) using AI to provide real-time risk assessment and automated alerts.

## 📊 System Architecture

### Data Sources (Multi-Source Fusion)
1. **Geotechnical Data**: Kaggle Slope Stability Dataset
   - Features: `slope_angle`, `pore_pressure`, `factor_of_safety`
   
2. **Environmental Data**: NASA Global Weather Data
   - Features: `rainfall_24h`, `temperature`, `humidity`, `soil_wetness`
   
3. **Visual Data**: Open Pit Mine Imagery
   - Features: `instability_score` (count of unstable regions)

### Core Features
- **Real-time Risk Prediction**: XGBoost model with 64.5% accuracy
- **Interactive Dashboard**: Streamlit-based visualization interface
- **Automated Alert System**: Email/SMS notifications for critical risks
- **Feature Scaling**: StandardScaler for normalized input processing

## 🛠️ Implementation Components

### 1. Data Fusion Pipeline (`data_fusion.py`)
- Creates "Virtual Mine" scenarios by sampling from all three data sources
- Generates 1000 training scenarios with logic-based risk labeling
- Handles missing data by creating realistic sample datasets

### 2. Machine Learning Model (`train_model_v2.py`)
- **Algorithm**: XGBoost Classifier
- **Features**: 4 core features (`slope_angle`, `pore_pressure`, `rainfall_24h`, `instability_score`)
- **Performance**: 64.5% accuracy, ROC AUC: 0.646
- **Outputs**: 
  - `rockfall_prediction_model.json` (primary model)
  - `feature_scaler.pkl` (scaling parameters)
  - `feature_columns.txt` (feature definitions)

### 3. Interactive Dashboard (`dashboard_v2.py`)
- **Real-time Controls**: Sliders for all 4 input features
- **Risk Visualization**: Gauge chart with color-coded risk levels
- **Alert Display**: Dynamic alerts based on risk thresholds
- **Feature Importance**: Visual representation of model decision factors
- **System Statistics**: Performance metrics and data source information

### 4. Alert System (`alert_system.py`)
- **Risk Thresholds**: 
  - Critical: ≥80% (immediate evacuation)
  - High: 60-79% (increased monitoring)
  - Medium: 40-59% (close monitoring)
  - Low: <40% (normal operations)
- **Notification Channels**: Email and SMS (configurable)
- **Alert History**: Logging and statistics tracking

## 📈 Model Performance

### Feature Importance Ranking
1. **Instability Score**: 44.71% (most influential)
2. **Rainfall (24h)**: 20.49%
3. **Pore Pressure**: 19.15%
4. **Slope Angle**: 15.65%

### Risk Classification Performance
- **Accuracy**: 64.5%
- **Precision**: 68% (High Risk), 63% (Low Risk)
- **Recall**: 43% (High Risk), 82% (Low Risk)
- **ROC AUC**: 0.646

## 🚀 How to Run the System

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Data Preparation
```bash
python data_fusion.py
```
- Creates `master_train.csv` with 1000 scenarios
- Generates sample datasets if real data not available

### Step 2: Model Training
```bash
python train_model_v2.py
```
- Trains XGBoost model on fused dataset
- Saves model artifacts for dashboard use

### Step 3: Launch Dashboard
```bash
streamlit run dashboard_v2.py
```
- Opens interactive dashboard at `http://localhost:8501`
- Provides real-time risk prediction interface

### Step 4: Test Alert System
```bash
python alert_system.py
```
- Tests alert functionality with sample scenarios
- Validates email/SMS notification system

## 🎛️ Dashboard Features

### Interactive Controls
- **Slope Angle**: 15-45° (mine wall steepness)
- **Pore Pressure**: 0.1-0.9 (ground water pressure)
- **Rainfall (24h)**: 0-100mm (weather trigger)
- **Instability Score**: 0-5 (visual crack detection)

### Risk Visualization
- **Gauge Chart**: Real-time risk percentage display
- **Color Coding**: Green (Low) → Yellow (Medium) → Orange (High) → Red (Critical)
- **Alert Messages**: Dynamic recommendations based on risk level

### System Information
- Data source details
- Model performance metrics
- Feature importance analysis
- Risk distribution statistics

## 🔧 Technical Specifications

### Model Architecture
- **Algorithm**: XGBoost (Gradient Boosting)
- **Objective**: Binary Classification (Risk/No Risk)
- **Features**: 4 normalized input parameters
- **Output**: Risk probability (0.0 - 1.0)

### Data Processing
- **Feature Scaling**: StandardScaler (Z-score normalization)
- **Train-Test Split**: 80% training, 20% testing
- **Cross-validation**: Stratified sampling for balanced classes

### Alert Logic
```python
if risk_probability >= 0.8:
    # Critical - Immediate evacuation
elif risk_probability >= 0.6:
    # High - Increased monitoring
elif risk_probability >= 0.4:
    # Medium - Close monitoring
else:
    # Low - Normal operations
```

## 📁 Project Structure
```
rockfall_prediction_tbp/
├── data_fusion.py              # Data merging and scenario generation
├── train_model_v2.py           # XGBoost model training
├── dashboard_v2.py             # Streamlit interactive dashboard
├── alert_system.py             # Automated alert notifications
├── requirements.txt            # Python dependencies
├── PROJECT_SUMMARY.md          # This documentation
├── master_train.csv            # Generated training dataset
├── rockfall_prediction_model.json  # Trained XGBoost model
├── feature_scaler.pkl          # Feature scaling parameters
└── feature_columns.txt         # Model feature definitions
```

## 🎯 Compliance with Requirements

### ✅ Multi-Source Data Integration
- Geotechnical (Kaggle): Internal mechanics
- Environmental (NASA): External triggers  
- Visual (Open Pit Mine): Surface evidence

### ✅ Real-time Prediction
- Live dashboard with interactive controls
- Instant risk probability calculation
- Dynamic risk level visualization

### ✅ Automated Alert System
- Multi-channel notifications (Email/SMS)
- Threshold-based alert triggering
- Alert history and statistics

### ✅ Scalable Architecture
- Modular component design
- Cross-platform model compatibility
- Easy integration with existing mine systems

## 🔮 Future Enhancements

1. **Real Data Integration**: Connect to actual mine sensors and weather APIs
2. **Advanced Models**: Implement deep learning for image analysis
3. **Mobile App**: Create mobile interface for field personnel
4. **Historical Analysis**: Add trend analysis and prediction forecasting
5. **Integration APIs**: Connect to existing mine management systems

## 📞 Support Information

**Dashboard Access**: http://localhost:8501
**Model Files**: `rockfall_prediction_model.json`
**Data Files**: `master_train.csv`
**Logs**: `rockfall_alerts.log`

---

*Built for NIRM/Ministry of Mines - Rockfall Prediction Challenge (PS 25061)*
*Multi-Source Data Fusion for Enhanced Mine Safety*
