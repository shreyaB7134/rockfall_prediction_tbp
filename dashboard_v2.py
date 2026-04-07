"""
Updated Streamlit Dashboard for Rockfall Prediction System
Integrates with the new XGBoost model and specific feature set
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
import joblib

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    .feature-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and feature scaler"""
    try:
        # Load XGBoost model
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model('rockfall_prediction_model.json')
        
        # Load feature scaler
        scaler = joblib.load('feature_scaler.pkl')
        
        # Load feature columns
        with open('feature_columns.txt', 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure train_model_v2.py has been run successfully.")
        return None, None, None

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
    """Create feature importance visualization based on model training"""
    
    # Use the actual feature importance from training results
    importance_data = [
        {'feature': 'Instability Score', 'importance': 0.4471},
        {'feature': 'Rainfall (24h)', 'importance': 0.2049},
        {'feature': 'Pore Pressure', 'importance': 0.1915},
        {'feature': 'Slope Angle', 'importance': 0.1565}
    ]
    
    df = pd.DataFrame(importance_data)
    
    fig = px.bar(df, x='importance', y='feature', orientation='h',
                title='Feature Importance in Rockfall Prediction',
                color='importance', color_continuous_scale='viridis',
                labels={'importance': 'Importance Score', 'feature': 'Features'})
    
    fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
    return fig

def create_risk_distribution_chart(df):
    """Create risk distribution visualization"""
    
    if df is None:
        return None
    
    # Create risk distribution
    risk_counts = df['risk_label'].value_counts()
    risk_labels = {0: 'Low Risk', 1: 'High Risk'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=[risk_labels.get(i, f'Class {i}') for i in risk_counts.index],
            y=risk_counts.values,
            marker_color=['green', 'red']
        )
    ])
    
    fig.update_layout(
        title='Risk Distribution in Training Data',
        xaxis_title='Risk Category',
        yaxis_title='Number of Scenarios',
        height=300
    )
    
    return fig

def predict_risk_probability(model, scaler, feature_columns, features):
    """Make risk prediction using the trained model"""
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure correct column order
    features_df = features_df[feature_columns]
    
    # Apply scaling
    features_scaled = scaler.transform(features_df)
    
    # Predict probability
    risk_probability = model.predict_proba(features_scaled)[0, 1]
    
    return risk_probability

def main():
    """Main dashboard application"""
    
    st.title("⛰️ Rockfall Prediction System")
    st.markdown("**Multi-Source Data Fusion for Mine Safety Monitoring**")
    st.markdown("---")
    
    # Load model and data
    model, scaler, feature_columns = load_model_and_scaler()
    historical_data = load_historical_data()
    
    if model is None:
        st.stop()
    
    # Sidebar for input controls
    st.sidebar.header("🎛️ Mine Conditions Panel")
    
    # Input sliders for the 4 key features
    st.sidebar.subheader("Current Measurements")
    
    slope_angle = st.sidebar.slider("Slope Angle (°)", 15.0, 45.0, 30.0, 0.5,
                                   help="Angle of the mine slope in degrees")
    
    pore_pressure = st.sidebar.slider("Pore Pressure", 0.1, 0.9, 0.5, 0.05,
                                     help="Ground water pressure (0.0-1.0 normalized)")
    
    rainfall_24h = st.sidebar.slider("Rainfall (24h)", 0.0, 100.0, 25.0, 1.0,
                                    help="Rainfall in last 24 hours in millimeters")
    
    instability_score = st.sidebar.slider("Instability Score", 0, 5, 1, 1,
                                        help="Number of unstable regions detected in imagery")
    
    # Predict button
    if st.sidebar.button("🔮 Predict Rockfall Risk", type="primary"):
        # Prepare features
        features = {
            'slope_angle': slope_angle,
            'pore_pressure': pore_pressure,
            'rainfall_24h': rainfall_24h,
            'instability_score': instability_score
        }
        
        # Make prediction
        risk_probability = predict_risk_probability(model, scaler, feature_columns, features)
        
        # Store in session state
        st.session_state.current_risk = risk_probability
        st.session_state.last_prediction = {
            'timestamp': datetime.now(),
            'risk_probability': risk_probability,
            'features': features
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
                st.error("**Recommended Actions:**")
                st.error("1. Immediate evacuation of affected areas")
                st.error("2. Deploy ground monitoring teams")
                st.error("3. Alert all personnel in vicinity")
                st.error("4. Implement emergency response protocols")
            elif risk >= 0.6:
                st.markdown('<div class="high-risk">⚠️ HIGH RISK - Increased Monitoring Required</div>', unsafe_allow_html=True)
                st.warning("**Recommended Actions:**")
                st.warning("1. Increase monitoring frequency")
                st.warning("2. Prepare evacuation plans")
                st.warning("3. Alert safety personnel")
            elif risk >= 0.4:
                st.markdown('<div class="medium-risk">⚡ MEDIUM RISK - Monitor Closely</div>', unsafe_allow_html=True)
                st.info("**Recommended Actions:**")
                st.info("1. Continue regular monitoring")
                st.info("2. Review safety procedures")
            else:
                st.markdown('<div class="low-risk">✅ LOW RISK - Normal Operations</div>', unsafe_allow_html=True)
                st.success("**Status:** Normal mining operations can continue")
        else:
            st.info("👈 Use the control panel to input current conditions and predict rockfall risk")
    
    with col2:
        st.header("📋 Current Conditions")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            features = pred['features']
            
            # Display current conditions with cards
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.metric("Slope Angle", f"{features['slope_angle']:.1f}°")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.metric("Pore Pressure", f"{features['pore_pressure']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.metric("Rainfall (24h)", f"{features['rainfall_24h']:.1f} mm")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.metric("Instability Score", f"{features['instability_score']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
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
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col4:
        st.header("📉 Risk Distribution")
        if historical_data is not None:
            fig_distribution = create_risk_distribution_chart(historical_data)
            if fig_distribution:
                st.plotly_chart(fig_distribution, use_container_width=True)
    
    # System information
    st.markdown("---")
    st.header("ℹ️ System Information")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.info("**Data Sources:**")
        st.info("• Kaggle Slope Stability")
        st.info("• NASA Weather Data")
        st.info("• Open Pit Mine Imagery")
    
    with col6:
        st.info("**Model Features:**")
        for feature in feature_columns:
            st.info(f"• {feature.replace('_', ' ').title()}")
    
    with col7:
        st.info("**Model Performance:**")
        st.info("• Accuracy: 64.5%")
        st.info("• ROC AUC: 0.646")
        st.info("• Algorithm: XGBoost")
    
    # Footer
    st.markdown("---")
    st.markdown("**Rockfall Prediction System** - Multi-source data fusion for mine safety")
    st.caption("Last system update: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    st.caption("Built for NIRM/Ministry of Mines - PS 25061 Rockfall Prediction Challenge")

if __name__ == "__main__":
    main()
