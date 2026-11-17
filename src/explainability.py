from pathlib import Path

import joblib
import pandas as pd

from preprocessing import (
    BASE_FEATURES,
    add_features
)


ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"


def explain_one(activity):

    import shap

    model = joblib.load(
        MODEL_DIR / "random_forest_model.joblib"
    )

    feature_names = joblib.load(
        MODEL_DIR / "feature_names.joblib"
    )

    row = pd.DataFrame([
        {
            col: activity[col]
            for col in BASE_FEATURES
        }
    ])

    for col in BASE_FEATURES:

        row[col] = pd.to_numeric(
            row[col],
            errors="coerce"
        )

        if row[col].isna().any() or row[col].iloc[0] < 0:
            raise ValueError(f"Invalid value for {col}")

    for col in ["login_after_hours", "usb_connected"]:

        if row[col].iloc[0] not in [0, 1]:
            raise ValueError(f"{col} must be 0 or 1.")

    row = add_features(row)

    row = row[feature_names]

    # SHAP explanation
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):

        values = (
            shap_values[1]
            if len(shap_values) > 1
            else shap_values[0]
        )

    else:

        values = shap_values

        if values.ndim == 3:
            values = values[:, :, 1]

    values = values[0]

    result = pd.DataFrame({

        "Feature": feature_names,

        "Input_Value": row.iloc[0].values,

        "SHAP_Value": values

    })

    result = result.reindex(
        result["SHAP_Value"]
        .abs()
        .sort_values(ascending=False)
        .index
    )

    return result.reset_index(drop=True)


if __name__ == "__main__":

    sample = {

        "login_after_hours": 1,

        "files_downloaded": 150,

        "usb_connected": 1,

        "failed_logins": 6,

        "sensitive_file_access": 12
    }

    explanation = explain_one(sample)

    print(explanation)