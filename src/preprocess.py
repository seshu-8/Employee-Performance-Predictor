"""
preprocess.py
-------------
Data Cleaning & Feature Engineering Pipeline
Prepares raw HR data for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path="data/employee_data.csv"):
    df = pd.read_csv(path)
    print(f"✅ Data loaded  → shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df):
    original_rows = len(df)

    # Drop duplicate employee records
    df = df.drop_duplicates(subset=["employee_id"])

    # Drop rows with missing values in key columns
    key_cols = ["age", "salary", "performance_score", "performance_label"]
    df = df.dropna(subset=key_cols)

    # Clamp numerical outliers to realistic HR ranges
    df["age"]               = df["age"].clip(18, 65)
    df["salary"]            = df["salary"].clip(25000, 250000)
    df["training_hours"]    = df["training_hours"].clip(0, 120)
    df["absenteeism_days"]  = df["absenteeism_days"].clip(0, 60)
    df["overtime_hours"]    = df["overtime_hours"].clip(0, 80)
    df["manager_rating"]    = df["manager_rating"].clip(1.0, 5.0)
    df["performance_score"] = df["performance_score"].clip(0, 100)

    print(f"✅ Cleaned data → {original_rows} → {len(df)} rows (removed {original_rows - len(df)} bad rows)")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    # Salary-to-experience ratio: are they well-compensated for experience?
    df["salary_per_exp_year"] = df["salary"] / (df["years_experience"] + 1)

    # Productivity index: projects per working month proxy
    df["productivity_index"] = (
        df["projects_completed"] / (df["years_experience"] + 1)
    ).round(3)

    # Engagement score: combines satisfaction + manager rating
    df["engagement_score"] = (
        (df["employee_satisfaction"] + df["manager_rating"]) / 2
    ).round(3)

    # Overwork flag: overtime > 20 hrs/month = risk of burnout
    df["overwork_flag"] = (df["overtime_hours"] > 20).astype(int)

    # Absenteeism flag: more than 10 days absent = concern
    df["high_absenteeism"] = (df["absenteeism_days"] > 10).astype(int)

    # Seniority level (numeric)
    level_map = {"Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Manager": 5}
    df["job_level_num"] = df["job_level"].map(level_map)

    print("✅ Feature engineering done → new columns added")
    return df


# ─────────────────────────────────────────────
# 4. ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────
def encode_features(df, fit=True, encoder_path="models/label_encoders.pkl"):
    categorical_cols = ["gender", "education_level", "department", "job_level"]

    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col])
            encoders[col] = le
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(encoders, encoder_path)
        print(f"✅ Encoders saved → {encoder_path}")
    else:
        encoders = joblib.load(encoder_path)
        for col in categorical_cols:
            df[col + "_enc"] = encoders[col].transform(df[col])
        print("✅ Encoders loaded and applied")

    # Encode target label
    target_map = {"Low": 0, "Medium": 1, "High": 2}
    df["target"] = df["performance_label"].map(target_map)

    return df, encoders


# ─────────────────────────────────────────────
# 5. SELECT FEATURES FOR MODEL
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "age", "years_experience", "salary", "training_hours",
    "absenteeism_days", "projects_completed", "overtime_hours",
    "manager_rating", "employee_satisfaction", "years_since_promotion",
    "salary_per_exp_year", "productivity_index", "engagement_score",
    "overwork_flag", "high_absenteeism", "job_level_num",
    "gender_enc", "education_level_enc", "department_enc"
]

def get_features_target(df):
    X = df[FEATURE_COLS]
    y = df["target"]
    return X, y


# ─────────────────────────────────────────────
# 6. SCALE FEATURES
# ─────────────────────────────────────────────
def scale_features(X_train, X_test, scaler_path="models/scaler.pkl"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved  → {scaler_path}")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# FULL PIPELINE (callable from main)
# ─────────────────────────────────────────────
def run_preprocessing(data_path="data/employee_data.csv"):
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_features(df)
    return df
