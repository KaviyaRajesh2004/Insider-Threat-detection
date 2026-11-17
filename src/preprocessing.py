import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "employee_id",
    "login_after_hours",
    "files_downloaded",
    "usb_connected",
    "failed_logins",
    "sensitive_file_access",
    "threat_label"
]

BASE_FEATURES = [
    "login_after_hours",
    "files_downloaded",
    "usb_connected",
    "failed_logins",
    "sensitive_file_access"
]

FEATURE_COLUMNS = BASE_FEATURES + [
    "total_activity_indicators"
]


def load_dataset(csv_path):

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def clean_dataset(df):

    data = df[REQUIRED_COLUMNS].copy()

    data = data.drop_duplicates()

    for col in BASE_FEATURES + ["threat_label"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    if data["threat_label"].isna().any():
        raise ValueError("Target labels cannot be missing.")

    for col in BASE_FEATURES:
        data[col] = data[col].fillna(
            data[col].median()
        )

    for col in ["login_after_hours", "usb_connected"]:
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} must contain 0 or 1.")

    for col in [
        "files_downloaded",
        "failed_logins",
        "sensitive_file_access"
    ]:
        if (data[col] < 0).any():
            raise ValueError(f"{col} cannot be negative.")

    if not data["threat_label"].isin([0, 1]).all():
        raise ValueError("Invalid threat labels.")

    data["employee_id"] = data["employee_id"].astype(str)

    data[BASE_FEATURES + ["threat_label"]] = (
        data[BASE_FEATURES + ["threat_label"]].astype(int)
    )

    return data


def add_features(df):

    data = df.copy()

    data["total_activity_indicators"] = (
        data["login_after_hours"]
        + data["usb_connected"]
        + (data["failed_logins"] >= 5).astype(int)
        + (data["files_downloaded"] >= 100).astype(int)
        + (data["sensitive_file_access"] >= 10).astype(int)
    )

    return data


def prepare_features(df):

    featured_data = add_features(df)

    X = featured_data[FEATURE_COLUMNS]

    y = featured_data["threat_label"]

    return X, y