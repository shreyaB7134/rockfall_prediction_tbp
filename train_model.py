"""
Training Script for Rockfall Prediction Model
Uses XGBoost for high accuracy on multi-source tabular data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

def load_data(filepath: str = 'master_train.csv') -> pd.DataFrame:
    """Load the master training dataset"""
    return pd.read_csv(filepath)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for training"""
    
    # Feature columns (exclude scenario_id and risk_label)
    feature_cols = [col for col in df.columns if col not in ['scenario_id', 'risk_label']]
    
    X = df[feature_cols]
    y = df['risk_label']
    
    return X, y

def train_xgboost_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Train XGBoost model with optimized parameters"""
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize XGBoost classifier with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train the model
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - Rockfall Prediction Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

def save_model(model: xgb.XGBClassifier, filepath: str = 'rockfall_model.json'):
    """Save trained model to file"""
    model.save_model(filepath)
    print(f"\nModel saved as '{filepath}'")

def save_model_pickle(model: xgb.XGBClassifier, filepath: str = 'rockfall_model.pkl'):
    """Save trained model as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as '{filepath}'")

def load_model(filepath: str = 'rockfall_model.json') -> xgb.XGBClassifier:
    """Load trained model from file"""
    model = xgb.XGBClassifier()
    model.load_model(filepath)
    return model

def predict_risk(model: xgb.XGBClassifier, features: Dict) -> float:
    """Predict rockfall risk for given features"""
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Ensure correct column order
    expected_features = ['slope_angle', 'pore_pressure', 'factor_of_safety', 
                        'rainfall', 'temperature', 'humidity', 'soil_wetness', 
                        'instability_score']
    
    for feature in expected_features:
        if feature not in feature_df.columns:
            feature_df[feature] = 0.0  # Default value
    
    feature_df = feature_df[expected_features]
    
    # Predict probability
    risk_probability = model.predict_proba(feature_df)[0, 1]
    
    return risk_probability

def main():
    """Main training pipeline"""
    
    print("Loading data...")
    df = load_data()
    
    print("Preparing features...")
    X, y = prepare_features(df)
    
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    print("Training model...")
    model = train_xgboost_model(X, y)
    
    # Save model
    save_model(model)
    save_model_pickle(model)
    
    # Test prediction function
    print("\nTesting prediction function...")
    test_features = {
        'slope_angle': 30.0,
        'pore_pressure': 0.8,
        'factor_of_safety': 0.9,
        'rainfall': 60.0,
        'temperature': 25.0,
        'humidity': 70.0,
        'soil_wetness': 0.6,
        'instability_score': 3
    }
    
    risk = predict_risk(model, test_features)
    print(f"Test prediction - Risk probability: {risk:.4f} ({risk*100:.1f}%)")
    
    print("\nTraining completed successfully!")
    
    return model

if __name__ == "__main__":
    model = main()
