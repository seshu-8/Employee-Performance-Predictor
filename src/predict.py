"""
predict.py
----------
Prediction Engine — Single Employee & Batch Prediction
Loads trained model and returns performance predictions with HR insights.
"""

import pandas as pd
import numpy as np
import joblib
import json

# ─────────────────────────────────────────────
# LABEL & INSIGHT MAPS
# ─────────────────────────────────────────────
LABEL_MAP    = {0: "Low", 1: "Medium", 2: "High"}
LABEL_EMOJI  = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}

HR_INSIGHTS = {
    "High": [
        "🏆 Promote to leadership or senior role — top performer.",
        "💡 Assign mentoring responsibilities to leverage expertise.",
        "📈 Consider fast-track career development programme.",
        "💰 Review compensation to ensure market alignment.",
    ],
    "Medium": [
        "📚 Enroll in targeted skills training programme.",
        "🤝 Schedule regular 1-on-1 coaching sessions with manager.",
        "🎯 Set clear KPIs and quarterly performance goals.",
        "🔍 Identify blockers preventing them from reaching High tier.",
    ],
    "Low": [
        "⚠️  Immediate performance improvement plan (PIP) required.",
        "🩺 Conduct a wellbeing check — may be disengaged or stressed.",
        "📋 Review absenteeism records and attendance patterns.",
        "🔄 Evaluate role fit — consider lateral move or additional support.",
        "📞 HR should arrange a confidential one-on-one meeting.",
    ],
}

FEATURE_COLS = [
    "age", "years_experience", "salary", "training_hours",
    "absenteeism_days", "projects_completed", "overtime_hours",
    "manager_rating", "employee_satisfaction", "years_since_promotion",
    "salary_per_exp_year", "productivity_index", "engagement_score",
    "overwork_flag", "high_absenteeism", "job_level_num",
    "gender_enc", "education_level_enc", "department_enc"
]


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts(model_path="models/best_model.pkl",
                   scaler_path="models/scaler.pkl",
                   encoder_path="models/label_encoders.pkl"):
    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    encoders = joblib.load(encoder_path)
    return model, scaler, encoders


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (mirror of preprocess.py)
# ─────────────────────────────────────────────
def _engineer(row: dict) -> dict:
    row["salary_per_exp_year"] = row["salary"] / (row["years_experience"] + 1)
    row["productivity_index"]  = round(row["projects_completed"] / (row["years_experience"] + 1), 3)
    row["engagement_score"]    = round((row["employee_satisfaction"] + row["manager_rating"]) / 2, 3)
    row["overwork_flag"]       = int(row["overtime_hours"] > 20)
    row["high_absenteeism"]    = int(row["absenteeism_days"] > 10)
    level_map = {"Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Manager": 5}
    row["job_level_num"]       = level_map.get(row["job_level"], 1)
    return row


def _encode(row: dict, encoders: dict) -> dict:
    for col in ["gender", "education_level", "department", "job_level"]:
        le = encoders[col]
        try:
            row[col + "_enc"] = int(le.transform([row[col]])[0])
        except ValueError:
            row[col + "_enc"] = 0   # unknown category → default
    return row


# ─────────────────────────────────────────────
# SINGLE EMPLOYEE PREDICTION
# ─────────────────────────────────────────────
def predict_single(employee: dict,
                   model=None, scaler=None, encoders=None):
    """
    Predict performance for one employee dict.

    Returns dict with:
      prediction_label, prediction_num, probabilities, hr_insights
    """
    if model is None:
        model, scaler, encoders = load_artifacts()

    emp = employee.copy()
    emp = _engineer(emp)
    emp = _encode(emp, encoders)

    X = pd.DataFrame([emp])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    pred_num  = int(model.predict(X_scaled)[0])
    pred_label = LABEL_MAP[pred_num]
    proba     = model.predict_proba(X_scaled)[0]

    proba_dict = {LABEL_MAP[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)}

    result = {
        "prediction_label"  : pred_label,
        "prediction_num"    : pred_num,
        "emoji"             : LABEL_EMOJI[pred_label],
        "probabilities_%"   : proba_dict,
        "hr_insights"       : HR_INSIGHTS[pred_label],
    }
    return result


# ─────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────
def predict_batch(df: pd.DataFrame,
                  model=None, scaler=None, encoders=None,
                  save_path="outputs/predictions.csv"):
    if model is None:
        model, scaler, encoders = load_artifacts()

    from src.preprocess import engineer_features, FEATURE_COLS as FC
    df_copy = df.copy()
    df_copy = engineer_features(df_copy)

    # encode
    for col in ["gender", "education_level", "department", "job_level"]:
        le = encoders[col]
        df_copy[col + "_enc"] = le.transform(df_copy[col])

    X = df_copy[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)

    df_copy["predicted_label"] = [LABEL_MAP[p] for p in preds]
    df_copy["prob_Low_%"]    = (probas[:, 0] * 100).round(1)
    df_copy["prob_Medium_%"] = (probas[:, 1] * 100).round(1)
    df_copy["prob_High_%"]   = (probas[:, 2] * 100).round(1)

    if save_path:
        import os
        os.makedirs("outputs", exist_ok=True)
        df_copy.to_csv(save_path, index=False)
        print(f"✅ Batch predictions saved → {save_path}")

    return df_copy


# ─────────────────────────────────────────────
# PRETTY PRINT SINGLE RESULT
# ─────────────────────────────────────────────
def display_result(employee: dict, result: dict):
    print("\n" + "="*60)
    print("  🏢  EMPLOYEE PERFORMANCE PREDICTION REPORT")
    print("="*60)
    print(f"  Employee ID   : {employee.get('employee_id', 'N/A')}")
    print(f"  Department    : {employee.get('department', 'N/A')}")
    print(f"  Job Level     : {employee.get('job_level', 'N/A')}")
    print(f"  Experience    : {employee.get('years_experience', 'N/A')} years")
    print("-"*60)
    print(f"  ⚡ Prediction  : {result['emoji']}  {result['prediction_label']} Performer")
    print(f"\n  📊 Confidence Scores:")
    for lbl, prob in result["probabilities_%"].items():
        bar = "█" * int(prob / 5)
        print(f"     {lbl:8s}: {prob:5.1f}%  {bar}")
    print(f"\n  💼 HR Action Items:")
    for tip in result["hr_insights"]:
        print(f"     • {tip}")
    print("="*60)


# ─────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Example employee (customize freely)
    sample_employee = {
        "employee_id"          : 9999,
        "age"                  : 34,
        "gender"               : "Male",
        "education_level"      : "Master's",
        "department"           : "Engineering",
        "job_level"            : "Senior",
        "years_experience"     : 9,
        "salary"               : 85000,
        "training_hours"       : 40,
        "absenteeism_days"     : 3,
        "projects_completed"   : 8,
        "overtime_hours"       : 12,
        "manager_rating"       : 4.2,
        "employee_satisfaction": 3.8,
        "years_since_promotion": 1,
    }

    result = predict_single(sample_employee)
    display_result(sample_employee, result)
