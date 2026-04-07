# Rockfall Prediction System

🏔️ **AI-powered rockfall risk prediction system for Indian mining regions**

## 🚀 Quick Start

### 1. Data Preparation
```bash
python data_fusion.py
```
*Loads weather data from 4 Indian mining regions and creates training dataset*

### 2. Launch Dashboard
```bash
streamlit run dashboard.py
```
*Opens interactive risk prediction dashboard*

### 3. Test Alert System
```bash
python alert_system.py
```
*Tests automated alert notifications*

## 📁 Project Structure

```
rockfall_prediction_tbp/
├── data/                          # 📊 All data files
│   ├── india_mining_regions_weather_combined.csv  # Weather data (10,228 records)
│   ├── weather_summary_statistics.csv  # Weather stats
│   ├── india_mining_regions_weather_comparison.png  # Weather plots
│   └── rockfall_alerts.log      # Alert system logs
│
├── data_fusion.py               # 🏗️ Weather data processing
├── dashboard.py                 # 📈 Interactive dashboard
├── alert_system.py              # 🚨 Alert notifications
├── requirements.txt             # 📦 Python dependencies
└── README.md                   # 📖 This file
```

## 🌍 Data Sources

### Indian Mining Regions (2018-2024)
- **Chhattisgarh** - Korba Coal Mines
- **Jharkhand** - Jharia Coalfield  
- **Karnataka** - Bellary Iron Ore Belt
- **Odisha** - Keonjhar Iron Ore Mines

### Features
- **Environmental**: rainfall_24h, temperature, humidity, soil_wetness (real NASA data)
- **Geotechnical**: *Add your slope_angle, pore_pressure, cohesion data*
- **Visual**: *Add your mine imagery for instability analysis*

## 🎯 System Capabilities

- ✅ **Multi-source data fusion** from weather, geotechnical, and visual sources
- ✅ **Real-time risk prediction** with probability scoring
- ✅ **Interactive dashboard** with regional weather data
- ✅ **Automated alerts** for critical risk scenarios
- ✅ **Regional analysis** across 4 Indian mining states

## 📊 Dataset Statistics

- **Weather Records**: 10,228 daily records (2018-2024)
- **Regions**: 4 major Indian mining areas
- **Date Range**: January 2018 to December 2024
- **Ready for**: Add your geotechnical and image data

## 🔧 Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## 📈 Model Performance

*Model will be trained when geotechnical and image data are added*
- **Expected Features**: Real weather + your geotechnical + visual data
- **Target**: High accuracy rockfall risk prediction
- **Cross-validation**: Will be validated across regions

## 🚨 Alert Thresholds

- **Critical**: ≥80% risk → Immediate evacuation
- **High**: 60-79% risk → Increased monitoring  
- **Medium**: 40-59% risk → Close monitoring
- **Low**: <40% risk → Normal operations

## 📞 Support

Built for NIRM/Ministry of Mines - Rockfall Prediction Challenge

---

**Total Files: 5 (Clean structure - ready for your data)**
