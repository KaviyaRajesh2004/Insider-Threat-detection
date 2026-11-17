from pathlib import Path

import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from preprocessing import (
    load_dataset,
    clean_dataset,
    prepare_features
)


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "dataset" / "employee_activity.csv"

MODEL_DIR = ROOT / "models"


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = load_dataset(DATA_PATH)

    # Clean dataset
    df = clean_dataset(df)

    # Prepare features
    X, y = prepare_features(df)

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain both normal and suspicious records."
        )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest Model
    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    random_forest.fit(X_train, y_train)

    # Isolation Forest Model
    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    normal_data = X_train[y_train == 0]

    isolation_forest.fit(normal_data)

    # Evaluation
    predictions = random_forest.predict(X_test)

    print("\nRandom Forest Evaluation")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(confusion_matrix(y_test, predictions))

    # Save models
    joblib.dump(
        random_forest,
        MODEL_DIR / "random_forest_model.joblib"
    )

    joblib.dump(
        isolation_forest,
        MODEL_DIR / "isolation_forest_model.joblib"
    )

    joblib.dump(
        list(X.columns),
        MODEL_DIR / "feature_names.joblib"
    )

    # Save test data
    test_data = X_test.copy()

    test_data["threat_label"] = y_test

    test_data.to_csv(
        MODEL_DIR / "test_data.csv",
        index=False
    )

    print("\nModels trained and saved successfully!")


if __name__ == "__main__":
    main()