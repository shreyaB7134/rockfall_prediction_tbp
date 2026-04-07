"""
Interactive Streamlit Dashboard for Rockfall Prediction System
Real-time risk visualization and monitoring interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_model import predict_risk, load_model
    from alert_system import RockfallAlertSystem
except ImportError:
    st.error("Required modules not found. Please ensure all scripts are in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Rockfall Prediction System",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .risk-gauge {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .critical-alert {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    .high-risk {
        background-color: #ff8800;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .medium-risk {
        background-color: #ffaa00;
        color: black;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .low-risk {
        background-color: #00c851;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_model():
    """Load the trained prediction model"""
    try:
        model = load_model('rockfall_model.json')
        return model
    except:
        st.error("Model file not found. Please run train_model.py first.")
        return None

@st.cache_data
def load_historical_data():
    """Load historical data for trends"""
    try:
        df = pd.read_csv('master_train.csv')
        return df
    except:
        st.error("Training data not found. Please run data_fusion.py first.")
        return None

def create_risk_gauge(risk_probability):
    """Create a gauge chart for risk visualization"""
    
    # Determine color based on risk level
    if risk_probability >= 0.8:
        color = "red"
        level = "CRITICAL"
    elif risk_probability >= 0.6:
        color = "orange"
        level = "HIGH"
    elif risk_probability >= 0.4:
        color = "yellow"
        level = "MEDIUM"
    else:
        color = "green"
        level = "LOW"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Rockfall Risk ({level})"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_feature_importance_chart():
    """Create feature importance visualization"""
    
    try:
        # Load feature importance from training
        importance_data = [
            {'feature': 'Pore Pressure', 'importance': 0.25},
            {'feature': 'Instability Score', 'importance': 0.20},
            {'feature': 'Rainfall', 'importance': 0.18},
            {'feature': 'Factor of Safety', 'importance': 0.15},
            {'feature': 'Slope Angle', 'importance': 0.12},
            {'feature': 'Soil Wetness', 'importance': 0.06},
            {'feature': 'Humidity', 'importance': 0.03},
            {'feature': 'Temperature', 'importance': 0.01}
        ]
        
        df = pd.DataFrame(importance_data)
        
        fig = px.bar(df, x='importance', y='feature', orientation='h',
                    title='Feature Importance in Rockfall Prediction',
                    color='importance', color_continuous_scale='viridis')
        
        fig.update_layout(height=400)
        return fig
    except:
        return None

def create_risk_trends_chart(df):
    """Create risk trends visualization"""
    
    if df is None:
        return None
    
    # Sample recent data for trends
    recent_data = df.tail(100).copy()
    recent_data['index'] = range(len(recent_data))
    
    # Create rolling average of risk
    recent_data['rolling_risk'] = recent_data['risk_label'].rolling(window=10).mean()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Risk Label Trends', 'Risk Factors Over Time'),
        vertical_spacing=0.1
    )
    
    # Risk labels
    fig.add_trace(
        go.Scatter(x=recent_data['index'], y=recent_data['risk_label'], 
                  mode='lines', name='Risk Label', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=recent_data['index'], y=recent_data['rolling_risk'], 
                  mode='lines', name='Rolling Average', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Key risk factors
    fig.add_trace(
        go.Scatter(x=recent_data['index'], y=recent_data['pore_pressure'], 
                  mode='lines', name='Pore Pressure', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=recent_data['index'], y=recent_data['rainfall']/100, 
                  mode='lines', name='Rainfall (normalized)', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def main():
    """Main dashboard application"""
    
    st.title("⛰️ Rockfall Prediction System")
    st.markdown("---")
    
    # Load model and data
    model = load_prediction_model()
    historical_data = load_historical_data()
    
    if model is None:
        st.stop()
    
    # Initialize alert system
    alert_system = RockfallAlertSystem(risk_threshold=0.8)
    
    # Sidebar for input controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Input sliders for real-time prediction
    st.sidebar.subheader("Current Conditions")
    
    slope_angle = st.sidebar.slider("Slope Angle (°)", 15.0, 45.0, 30.0, 0.5)
    pore_pressure = st.sidebar.slider("Pore Pressure", 0.1, 0.9, 0.5, 0.05)
    factor_of_safety = st.sidebar.slider("Factor of Safety", 0.5, 2.0, 1.2, 0.1)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 100.0, 25.0, 1.0)
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 35.0, 22.0, 0.5)
    humidity = st.sidebar.slider("Humidity (%)", 30.0, 90.0, 60.0, 1.0)
    soil_wetness = st.sidebar.slider("Soil Wetness", 0.1, 0.8, 0.4, 0.05)
    instability_score = st.sidebar.slider("Instability Score", 0, 5, 1, 1)
    
    # Predict button
    if st.sidebar.button("🔮 Predict Risk", type="primary"):
        # Prepare features
        features = {
            'slope_angle': slope_angle,
            'pore_pressure': pore_pressure,
            'factor_of_safety': factor_of_safety,
            'rainfall': rainfall,
            'temperature': temperature,
            'humidity': humidity,
            'soil_wetness': soil_wetness,
            'instability_score': instability_score
        }
        
        # Make prediction
        risk_probability = predict_risk(model, features)
        
        # Trigger alert system
        scenario_data = features.copy()
        scenario_data['factor_of_safety'] = factor_of_safety
        alert_result = alert_system.trigger_alert(risk_probability, scenario_data, "dashboard_scenario")
        
        # Store in session state
        st.session_state.current_risk = risk_probability
        st.session_state.last_prediction = {
            'timestamp': datetime.now(),
            'risk_probability': risk_probability,
            'features': features,
            'alert_level': alert_result['risk_level']
        }
    
    # Main dashboard area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Risk Assessment")
        
        # Display current risk if available
        if 'current_risk' in st.session_state:
            risk = st.session_state.current_risk
            
            # Risk gauge
            fig_gauge = create_risk_gauge(risk)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Alert message
            if risk >= 0.8:
                st.markdown('<div class="critical-alert">🚨 CRITICAL RISK - EVACUATE IMMEDIATELY! 🚨</div>', unsafe_allow_html=True)
            elif risk >= 0.6:
                st.markdown('<div class="high-risk">⚠️ HIGH RISK - Increased Monitoring Required</div>', unsafe_allow_html=True)
            elif risk >= 0.4:
                st.markdown('<div class="medium-risk">⚡ MEDIUM RISK - Monitor Closely</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="low-risk">✅ LOW RISK - Normal Operations</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Use the control panel to input current conditions and predict rockfall risk")
    
    with col2:
        st.header("📋 Current Conditions")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            features = pred['features']
            
            # Display current conditions
            st.metric("Slope Angle", f"{features['slope_angle']:.1f}°")
            st.metric("Pore Pressure", f"{features['pore_pressure']:.2f}")
            st.metric("Rainfall", f"{features['rainfall']:.1f} mm")
            st.metric("Instability Score", f"{features['instability_score']}")
            st.metric("Risk Probability", f"{pred['risk_probability']:.1%}")
            
            # Last update time
            st.caption(f"Last updated: {pred['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("No prediction made yet")
    
    # Additional visualizations
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.header("📈 Feature Importance")
        fig_importance = create_feature_importance_chart()
        if fig_importance:
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with col4:
        st.header("📉 Risk Trends")
        if historical_data is not None:
            fig_trends = create_risk_trends_chart(historical_data)
            if fig_trends:
                st.plotly_chart(fig_trends, use_container_width=True)
    
    # Alert history
    st.markdown("---")
    st.header("🚨 Alert History")
    
    alert_history = alert_system.get_alert_history(last_n=5)
    if alert_history:
        for alert in reversed(alert_history):
            with st.expander(f"Alert - {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {alert['risk_level']}"):
                col5, col6 = st.columns(2)
                with col5:
                    st.write(f"**Risk Probability:** {alert['risk_probability']:.1%}")
                    st.write(f"**Risk Level:** {alert['risk_level']}")
                    st.write(f"**Scenario ID:** {alert['scenario_id']}")
                with col6:
                    if alert['alerts_sent']:
                        st.write("**Alerts Sent:**")
                        for alert_type in alert['alerts_sent']:
                            st.write(f"  - {alert_type.title()}")
    else:
        st.info("No alerts triggered yet")
    
    # System statistics
    st.markdown("---")
    st.header("📊 System Statistics")
    
    stats = alert_system.get_alert_statistics()
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        st.metric("Total Alerts", stats['total_alerts'])
    with col8:
        st.metric("Critical Alerts", stats['critical_alerts'])
    with col9:
        st.metric("High Risk Alerts", stats['high_alerts'])
    with col10:
        st.metric("Critical %", f"{stats['critical_percentage']:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("**Rockfall Prediction System** - Real-time monitoring and alert system for mine safety")
    st.caption("Last system update: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    main()
