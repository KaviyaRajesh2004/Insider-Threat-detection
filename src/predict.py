from pathlib import Path

import joblib
import pandas as pd

from preprocessing import (
    BASE_FEATURES,
    add_features
)


ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"


class ThreatPredictor:

    def __init__(self):

        self.random_forest = joblib.load(
            MODEL_DIR / "random_forest_model.joblib"
        )

        self.isolation_forest = joblib.load(
            MODEL_DIR / "isolation_forest_model.joblib"
        )

        self.feature_names = joblib.load(
            MODEL_DIR / "feature_names.joblib"
        )

    def predict_one(self, activity):

        missing = [
            col for col in BASE_FEATURES
            if col not in activity
        ]

        if missing:
            raise ValueError(f"Missing fields: {missing}")

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

            if row[col].isna().any():
                raise ValueError(f"Invalid value for {col}")

            if row[col].iloc[0] < 0:
                raise ValueError(f"{col} cannot be negative.")

        for col in ["login_after_hours", "usb_connected"]:

            if row[col].iloc[0] not in [0, 1]:
                raise ValueError(f"{col} must be 0 or 1.")

        # Feature engineering
        row = add_features(row)

        row = row[self.feature_names]

        # Random Forest prediction
        rf_prediction = int(
            self.random_forest.predict(row)[0]
        )

        probabilities = self.random_forest.predict_proba(row)[0]

        probability_map = dict(
            zip(
                self.random_forest.classes_,
                probabilities
            )
        )

        suspicious_probability = float(
            probability_map.get(1, 0.0)
        )

        # Isolation Forest prediction
        isolation_prediction = int(
            self.isolation_forest.predict(row)[0]
        )

        anomaly = int(isolation_prediction == -1)

        return {
            "prediction": rf_prediction,
            "suspicious_probability": round(
                suspicious_probability, 4
            ),
            "anomaly": anomaly,
            "review_recommended": bool(
                rf_prediction == 1 or anomaly == 1
            )
        }


if __name__ == "__main__":

    sample_activity = {

        "login_after_hours": 1,

        "files_downloaded": 150,

        "usb_connected": 1,

        "failed_logins": 6,

        "sensitive_file_access": 12
    }

    predictor = ThreatPredictor()

    result = predictor.predict_one(sample_activity)

    print(result)