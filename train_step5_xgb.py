import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    df = pd.read_csv("data/final_rockfall_master_10k.csv")

    features = [
        "rainfall_mm",
        "rainfall_3d_avg",
        "humidity_percent",
        "soil_wetness",
        "adjusted_pore_pressure",
        "base_fs",
        "slope_angle",
        "cohesion_kpa",
        "instability_score",
    ]

    missing = [c for c in features + ["final_risk_label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[features]
    y = df["final_risk_label"].astype(int)

    print("Class distribution (final_risk_label):")
    print(y.value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rockfall_xgb_model.pkl")
    print("Saved model -> models/rockfall_xgb_model.pkl")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=features)
    plt.title("Feature Importance (Multi-Source Integration Proof)")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=200)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: Prediction Accuracy")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve: Model Reliability")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=200)
    plt.close()

    print("\nSaved plots -> models/feature_importance.png, models/confusion_matrix.png, models/roc_curve.png")


if __name__ == "__main__":
    main()
