import argparse

import joblib
import pandas as pd


FEATURES = [
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


def main():
    parser = argparse.ArgumentParser(description="Rockfall risk predictor (XGBoost)")
    parser.add_argument("--model", default="models/rockfall_xgb_model.pkl")

    parser.add_argument("--rainfall_mm", type=float, required=True)
    parser.add_argument("--rainfall_3d_avg", type=float, required=True)
    parser.add_argument("--humidity_percent", type=float, required=True)
    parser.add_argument("--soil_wetness", type=float, required=True)

    parser.add_argument("--adjusted_pore_pressure", type=float, required=True)
    parser.add_argument("--base_fs", type=float, required=True)
    parser.add_argument("--slope_angle", type=float, required=True)
    parser.add_argument("--cohesion_kpa", type=float, required=True)

    parser.add_argument("--instability_score", type=float, required=True)

    args = parser.parse_args()

    model = joblib.load(args.model)

    row = {
        "rainfall_mm": args.rainfall_mm,
        "rainfall_3d_avg": args.rainfall_3d_avg,
        "humidity_percent": args.humidity_percent,
        "soil_wetness": args.soil_wetness,
        "adjusted_pore_pressure": args.adjusted_pore_pressure,
        "base_fs": args.base_fs,
        "slope_angle": args.slope_angle,
        "cohesion_kpa": args.cohesion_kpa,
        "instability_score": args.instability_score,
    }

    X = pd.DataFrame([row], columns=FEATURES)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    print(f"Rockfall probability: {proba:.3f}")
    if pred == 1:
        print("ALERT: HIGH RISK")
    else:
        print("Status: LOW RISK")


if __name__ == "__main__":
    main()
