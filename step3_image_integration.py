import os
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Step3Config:
    annotations_root: str
    images_root: Optional[str] = None
    weather_geotech_csv: str = "data/weather_geotech_combined.csv"
    out_image_scores_csv: str = "data/image_instability_scores.csv"
    out_master_csv: str = "data/final_rockfall_master_10k.csv"


def _iter_json_files(root: str) -> List[str]:
    json_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                json_files.append(os.path.join(dirpath, fn))
    return sorted(json_files)


def _safe_int_label(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def compute_instability_score_from_labelme(json_path: str) -> int:
    """Compute instability_score for a single LabelMe-style JSON.

    Current dataset uses labels "0" and "1" in `shapes[*].label`.
    We treat label==1 as instability evidence and count its boxes.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    score = 0
    for sh in shapes:
        label = _safe_int_label(sh.get("label"))
        if label == 1:
            score += 1
    return score


def build_image_scores_df(annotations_root: str) -> pd.DataFrame:
    print(f"🖼️ Scanning annotations: {annotations_root}")
    json_files = _iter_json_files(annotations_root)
    if not json_files:
        raise FileNotFoundError(f"No .json annotations found under: {annotations_root}")

    rows = []
    for p in json_files:
        score = compute_instability_score_from_labelme(p)
        image_id = os.path.splitext(os.path.basename(p))[0]
        rows.append(
            {
                "image_id": image_id,
                "annotation_path": p,
                "instability_score": int(score),
            }
        )

    df = pd.DataFrame(rows)
    print(f"✅ Computed instability_score for {len(df):,} annotations")
    print(f"📊 instability_score distribution: {df['instability_score'].value_counts().sort_index().head(10).to_dict()}")
    return df


def align_and_merge(master_df: pd.DataFrame, image_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically align image scores to numerical scenarios.

    - Numerical scenarios are sorted safe->risky using base_fs (descending)
    - Image scores are sorted clean->cracked using instability_score (ascending)
    - Then we stretch images to scenario count using np.linspace indices.
    """

    if "base_fs" not in master_df.columns:
        raise ValueError("Expected column `base_fs` in weather_geotech_combined.csv")

    master_sorted = master_df.sort_values(by=["base_fs", "date", "region"], ascending=[False, True, True]).reset_index(drop=True)
    images_sorted = image_scores_df.sort_values(by=["instability_score", "image_id"], ascending=[True, True]).reset_index(drop=True)

    indices = np.linspace(0, len(images_sorted) - 1, len(master_sorted))
    indices = np.round(indices).astype(int)

    master_sorted["image_id"] = images_sorted.iloc[indices]["image_id"].values
    master_sorted["instability_score"] = images_sorted.iloc[indices]["instability_score"].values

    # Optional refined label that includes visual trigger
    # Keep existing `risk_label` from Step 2; define `final_risk_label` for Step 3.
    if "risk_label" in master_sorted.columns:
        base_risk = master_sorted["risk_label"].astype(int)
    else:
        base_risk = pd.Series(np.zeros(len(master_sorted), dtype=int))

    master_sorted["final_risk_label"] = np.where(
        (base_risk == 1)
        | ((master_sorted["base_fs"] < 1.1) & (master_sorted["instability_score"] > 0))
        | (master_sorted["instability_score"] >= 2),
        1,
        0,
    ).astype(int)

    return master_sorted


def main():
    config = Step3Config(
        annotations_root=r"C:\Users\akshi\Downloads\Open-Pit-Mine-Object-Detection-Dataset\Open-Pit-Mine-Object-Detection-Dataset\annotation",
        images_root=r"C:\Users\akshi\Downloads\Open-Pit-Mine-Object-Detection-Dataset\Open-Pit-Mine-Object-Detection-Dataset\images",
    )

    master_df = pd.read_csv(config.weather_geotech_csv)
    image_scores_df = build_image_scores_df(config.annotations_root)

    # Save intermediate scores
    os.makedirs(os.path.dirname(config.out_image_scores_csv), exist_ok=True)
    image_scores_df.to_csv(config.out_image_scores_csv, index=False)
    print(f"💾 Saved image scores: {config.out_image_scores_csv}")

    fused_master = align_and_merge(master_df, image_scores_df)
    fused_master.to_csv(config.out_master_csv, index=False)
    print(f"💾 Saved final master dataset: {config.out_master_csv}")

    print("\n📈 Final label distribution:")
    for col in ["risk_label", "final_risk_label"]:
        if col in fused_master.columns:
            print(f"- {col}: {fused_master[col].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
