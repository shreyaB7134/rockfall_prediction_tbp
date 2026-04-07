"""
Updated Training Script for Rockfall Prediction Model
Follows the specific feature fusion strategy: slope_angle, pore_pressure, rainfall_24h, instability_score
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(filepath='master_train.csv'):
    """Load and prepare the master dataset with specific features"""
    
    # Load the master dataset
    df = pd.read_csv(filepath)
    
    # Select the specific features as per the fusion strategy
    # Features: slope_angle, pore_pressure, rainfall_24h, instability_score
    feature_columns = ['slope_angle', 'pore_pressure', 'rainfall', 'instability_score']
    
    # Ensure we have all required columns
    available_columns = df.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Map rainfall to rainfall_24h if needed
    if 'rainfall' in df.columns and 'rainfall_24h' not in df.columns:
        df = df.rename(columns={'rainfall': 'rainfall_24h'})
        feature_columns = ['slope_angle', 'pore_pressure', 'rainfall_24h', 'instability_score']
    
    # Extract features and target
    X = df[feature_columns]
    y = df['risk_label']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Risk percentage: {y.mean() * 100:.1f}%")
    
    return X, y, feature_columns

def apply_feature_scaling(X_train, X_test):
    """Apply StandardScaler to ensure features are on the same scale"""
    
    scaler = StandardScaler()
    
    # Fit on training data and transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save the scaler for use in dashboard
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("Feature scaling applied and scaler saved as 'feature_scaler.pkl'")
    
    return X_train_scaled, X_test_scaled, scaler

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier with optimized parameters"""
    
    # Initialize XGBoost with parameters optimized for binary classification
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance - Rockfall Prediction Model')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, y_pred_proba

def save_model_and_artifacts(model, feature_columns):
    """Save model and related artifacts for dashboard use"""
    
    # Save model as JSON (cross-platform compatible)
    model.save_model('rockfall_prediction_model.json')
    print("Model saved as 'rockfall_prediction_model.json'")
    
    # Save model as pickle (backup)
    joblib.dump(model, 'rockfall_prediction_model.pkl')
    print("Model saved as 'rockfall_prediction_model.pkl'")
    
    # Save feature columns for dashboard
    with open('feature_columns.txt', 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    print("Feature columns saved as 'feature_columns.txt'")

def test_prediction_functionality(model, scaler, feature_columns):
    """Test the prediction functionality with sample scenarios"""
    
    print(f"\n{'='*50}")
    print("TESTING PREDICTION FUNCTIONALITY")
    print(f"{'='*50}")
    
    # Test scenarios representing different risk levels
    test_scenarios = [
        {
            'name': 'Low Risk Scenario',
            'features': {
                'slope_angle': 20.0,
                'pore_pressure': 0.3,
                'rainfall_24h': 10.0,
                'instability_score': 0
            }
        },
        {
            'name': 'Medium Risk Scenario',
            'features': {
                'slope_angle': 30.0,
                'pore_pressure': 0.6,
                'rainfall_24h': 40.0,
                'instability_score': 2
            }
        },
        {
            'name': 'High Risk Scenario',
            'features': {
                'slope_angle': 40.0,
                'pore_pressure': 0.8,
                'rainfall_24h': 80.0,
                'instability_score': 4
            }
        }
    ]
    
    for scenario in test_scenarios:
        # Convert features to DataFrame
        features_df = pd.DataFrame([scenario['features']])
        
        # Ensure correct column order
        features_df = features_df[feature_columns]
        
        # Apply scaling
        features_scaled = scaler.transform(features_df)
        
        # Predict probability
        risk_probability = model.predict_proba(features_scaled)[0, 1]
        
        print(f"\n{scenario['name']}:")
        print(f"  Features: {scenario['features']}")
        print(f"  Risk Probability: {risk_probability:.4f} ({risk_probability*100:.1f}%)")
        
        # Determine risk level
        if risk_probability >= 0.8:
            print(f"  Risk Level: CRITICAL 🚨")
        elif risk_probability >= 0.6:
            print(f"  Risk Level: HIGH ⚠️")
        elif risk_probability >= 0.4:
            print(f"  Risk Level: MEDIUM ⚡")
        else:
            print(f"  Risk Level: LOW ✅")

def main():
    """Main training pipeline following the specified fusion strategy"""
    
    print("🏗️  Rockfall Prediction Model Training")
    print("Following Multi-Source Fusion Strategy")
    print("Features: slope_angle, pore_pressure, rainfall_24h, instability_score")
    print("="*60)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    X, y, feature_columns = load_and_prepare_data()
    
    # Step 2: Train-test split
    print("\n🔀 Step 2: Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Step 3: Apply feature scaling
    print("\n⚖️  Step 3: Applying feature scaling...")
    X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(X_train, X_test)
    
    # Step 4: Train XGBoost model
    print("\n🤖 Step 4: Training XGBoost model...")
    model, y_pred_proba = train_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Step 5: Save model and artifacts
    print("\n💾 Step 5: Saving model and artifacts...")
    save_model_and_artifacts(model, feature_columns)
    
    # Step 6: Test prediction functionality
    print("\n🧪 Step 6: Testing prediction functionality...")
    test_prediction_functionality(model, scaler, feature_columns)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("Model ready for integration with Streamlit dashboard")
    print("="*60)
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    model, scaler, feature_columns = main()
