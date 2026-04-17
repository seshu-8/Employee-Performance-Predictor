"""
generate_data.py
----------------
Synthetic HR Dataset Generator for Employee Performance Predictor
Simulates realistic company HR data for model training and analysis.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
NUM_EMPLOYEES = 1000
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

DEPARTMENTS  = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations", "Support"]
GENDERS      = ["Male", "Female", "Non-Binary"]
EDUCATION    = ["High School", "Bachelor's", "Master's", "PhD"]
JOB_LEVELS   = ["Junior", "Mid", "Senior", "Lead", "Manager"]

# ─────────────────────────────────────────────
# HELPER – REALISTIC PERFORMANCE SCORE FORMULA
# ─────────────────────────────────────────────
def compute_performance(row):
    """
    Performance score (0-100) influenced by multiple HR factors.
    Mirrors how real companies build composite KPIs.
    """
    score = 50.0  # baseline

    # Experience boosts performance (diminishing returns)
    score += min(row["years_experience"] * 1.5, 20)

    # More training hours → better performance
    score += min(row["training_hours"] * 0.4, 10)

    # High absenteeism hurts performance
    score -= row["absenteeism_days"] * 1.2

    # Projects completed is a strong signal
    score += min(row["projects_completed"] * 1.8, 18)

    # Manager rating (1-5) carries weight
    score += (row["manager_rating"] - 3) * 4

    # Overtime is double-edged: moderate = good, excessive = burnout
    if row["overtime_hours"] < 10:
        score += row["overtime_hours"] * 0.5
    else:
        score -= (row["overtime_hours"] - 10) * 0.8

    # Salary-to-role alignment proxy (higher salary at junior level = overpaid risk)
    if row["job_level"] == "Junior" and row["salary"] > 70000:
        score -= 3

    # Education premium
    edu_map = {"High School": 0, "Bachelor's": 2, "Master's": 4, "PhD": 5}
    score += edu_map.get(row["education_level"], 0)

    # Clamp to [0, 100] and add Gaussian noise for realism
    score += np.random.normal(0, 4)
    return round(float(np.clip(score, 0, 100)), 2)


def classify_performance(score):
    """Label into three HR-standard tiers."""
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────
# MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────
def generate_dataset(n=NUM_EMPLOYEES, save_path="data/employee_data.csv"):
    records = []

    for emp_id in range(1001, 1001 + n):
        dept        = np.random.choice(DEPARTMENTS)
        gender      = np.random.choice(GENDERS, p=[0.55, 0.40, 0.05])
        education   = np.random.choice(EDUCATION, p=[0.10, 0.50, 0.30, 0.10])
        job_level   = np.random.choice(JOB_LEVELS, p=[0.25, 0.30, 0.25, 0.12, 0.08])
        age         = int(np.random.normal(35, 8))
        age         = max(22, min(60, age))

        # Experience correlated with age
        max_exp     = age - 21
        yrs_exp     = int(np.random.uniform(0, max_exp))

        # Salary influenced by level + experience
        base_salary = {"Junior": 40000, "Mid": 55000, "Senior": 75000,
                       "Lead": 95000, "Manager": 110000}[job_level]
        salary      = int(base_salary + yrs_exp * 1200 + np.random.normal(0, 5000))
        salary      = max(30000, salary)

        training_hrs   = int(np.random.exponential(20))
        training_hrs   = min(training_hrs, 80)

        absent_days    = int(np.random.exponential(4))
        absent_days    = min(absent_days, 30)

        projects       = int(np.random.normal(5, 2))
        projects       = max(0, min(15, projects))

        overtime_hrs   = int(np.random.exponential(8))
        overtime_hrs   = min(overtime_hrs, 50)

        mgr_rating     = round(np.random.normal(3.2, 0.8), 1)
        mgr_rating     = max(1.0, min(5.0, mgr_rating))

        satisfaction   = round(np.random.uniform(1, 5), 1)
        last_promo_yrs = int(np.random.exponential(2.5))
        last_promo_yrs = min(last_promo_yrs, 10)

        row = {
            "employee_id"         : emp_id,
            "age"                 : age,
            "gender"              : gender,
            "education_level"     : education,
            "department"          : dept,
            "job_level"           : job_level,
            "years_experience"    : yrs_exp,
            "salary"              : salary,
            "training_hours"      : training_hrs,
            "absenteeism_days"    : absent_days,
            "projects_completed"  : projects,
            "overtime_hours"      : overtime_hrs,
            "manager_rating"      : mgr_rating,
            "employee_satisfaction": satisfaction,
            "years_since_promotion": last_promo_yrs,
        }

        row["performance_score"] = compute_performance(row)
        row["performance_label"] = classify_performance(row["performance_score"])

        records.append(row)

    df = pd.DataFrame(records)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✅ Dataset saved → {save_path}")
    print(f"   Shape  : {df.shape}")
    print(f"   Labels : {df['performance_label'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
