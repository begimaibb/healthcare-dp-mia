import argparse
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

TARGET = "diabetes_diagnosis"
DROP_COLS = ["patient_id", TARGET]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def train(data_path: str, output_dir: str = "models/") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} records, {df[TARGET].mean():.1%} positive rate")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n--- Training ---")
    print(f"  Train size : {len(X_train)}")
    print(f"  Test size  : {len(X_test)}")

    model = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=500,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- Model Performance ---")
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, os.path.join(output_dir, "disease_risk_model.pkl"))
    joblib.dump(feature_cols, os.path.join(output_dir, "feature_names.pkl"))
    joblib.dump((X_test, y_test), os.path.join(output_dir, "test_data.pkl"))

    print(f"\nArtifacts saved to {output_dir}/")
    return {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_patient_data.csv")
    parser.add_argument("--output-dir", default="models/")
    args = parser.parse_args()
    train(args.data, args.output_dir)