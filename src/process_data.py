import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def calculate_age(birthdate_str: str) -> int:
    birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d")
    today = datetime.today()
    return (today - birthdate).days // 365


def extract_observation(entries: list, code: str):
    values = []
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue
        codings = resource.get("code", {}).get("coding", [])
        if any(c.get("code") == code for c in codings):
            val = resource.get("valueQuantity", {}).get("value")
            if val is not None:
                values.append(float(val))
    return float(np.mean(values)) if values else None


def parse_fhir_bundle(filepath: str):
    with open(filepath, "r") as f:
        bundle = json.load(f)
    entries = bundle.get("entry", [])
    record = {}

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            record["patient_id"] = resource.get("id", "unknown")
            birthdate = resource.get("birthDate")
            record["age"] = calculate_age(birthdate) if birthdate else np.nan
            gender = resource.get("gender", "unknown").lower()
            record["gender_male"] = 1 if gender == "male" else 0
            record["gender_female"] = 1 if gender == "female" else 0
            break

    if "patient_id" not in record:
        return None

    record["latest_bmi"] = extract_observation(entries, "39156-5")
    record["latest_a1c"] = extract_observation(entries, "4548-4")

    has_diabetes = False
    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            codings = resource.get("code", {}).get("coding", [])
            for c in codings:
                if c.get("code", "").startswith("E11") or "diabetes" in c.get("display", "").lower():
                    has_diabetes = True
    record["diabetes_diagnosis"] = int(has_diabetes)
    return record


def process_directory(input_dir: str, output_path: str):
    records = []
    files = list(Path(input_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")
    print(f"Processing {len(files)} FHIR bundles...")
    for fp in files:
        record = parse_fhir_bundle(str(fp))
        if record:
            records.append(record)
    df = pd.DataFrame(records)
    for col in ["latest_bmi", "latest_a1c", "age"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    print(f"Processed patients: {len(df)}")
    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthea/")
    parser.add_argument("--output", default="data/processed_patient_data.csv")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_directory(args.input, args.output)