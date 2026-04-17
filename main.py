"""
main.py
-------
🚀 Employee Performance Predictor — Main Entry Point
Orchestrates the full pipeline: data → model → predictions → report.

Usage:
    python main.py                  # run full pipeline
    python main.py --phase generate # only generate data
    python main.py --phase eda      # only run EDA
    python main.py --phase train    # only train model
    python main.py --phase predict  # only predict (needs trained model)
"""

import argparse
import time
import os
import sys

# Ensure src/ is importable regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       EMPLOYEE PERFORMANCE PREDICTOR  v1.0               ║
║       Data Analytics  |  Machine Learning  |  HR Insights║
╚══════════════════════════════════════════════════════════╝
"""


def phase_generate():
    print("\n📁 PHASE 1 — Generating Synthetic HR Dataset …")
    from src.generate_data import generate_dataset
    df = generate_dataset(n=1000, save_path="data/employee_data.csv")
    return df


def phase_eda(df=None):
    print("\n📊 PHASE 2 — Exploratory Data Analysis …")
    from src.preprocess import load_data, clean_data, engineer_features, encode_features
    from src.eda import run_eda

    if df is None:
        df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_features(df)
    run_eda(df)
    return df


def phase_train():
    print("\n🤖 PHASE 3 — Training Machine Learning Models …")
    from src.train_model import run_training
    model, results = run_training()
    return model, results


def phase_predict():
    print("\n🔮 PHASE 4 — Running Sample Predictions …")
    from src.predict import predict_single, display_result

    # ── Demo employees (edge cases) ──────────────────────────
    demo_employees = [
        {
            "name"                 : "Alice (High Performer)",
            "employee_id"          : 9001,
            "age"                  : 32,
            "gender"               : "Female",
            "education_level"      : "Master's",
            "department"           : "Engineering",
            "job_level"            : "Senior",
            "years_experience"     : 8,
            "salary"               : 90000,
            "training_hours"       : 55,
            "absenteeism_days"     : 2,
            "projects_completed"   : 10,
            "overtime_hours"       : 10,
            "manager_rating"       : 4.8,
            "employee_satisfaction": 4.5,
            "years_since_promotion": 1,
        },
        {
            "name"                 : "Bob (Medium Performer)",
            "employee_id"          : 9002,
            "age"                  : 40,
            "gender"               : "Male",
            "education_level"      : "Bachelor's",
            "department"           : "Sales",
            "job_level"            : "Mid",
            "years_experience"     : 10,
            "salary"               : 58000,
            "training_hours"       : 18,
            "absenteeism_days"     : 7,
            "projects_completed"   : 5,
            "overtime_hours"       : 16,
            "manager_rating"       : 3.2,
            "employee_satisfaction": 3.0,
            "years_since_promotion": 3,
        },
        {
            "name"                 : "Carol (At-Risk / Low Performer)",
            "employee_id"          : 9003,
            "age"                  : 27,
            "gender"               : "Female",
            "education_level"      : "High School",
            "department"           : "Support",
            "job_level"            : "Junior",
            "years_experience"     : 2,
            "salary"               : 35000,
            "training_hours"       : 5,
            "absenteeism_days"     : 18,
            "projects_completed"   : 1,
            "overtime_hours"       : 30,
            "manager_rating"       : 2.1,
            "employee_satisfaction": 1.8,
            "years_since_promotion": 0,
        },
    ]

    for emp in demo_employees:
        name = emp.pop("name")
        print(f"\n{'─'*60}")
        print(f"  Testing employee profile: {name}")
        result = predict_single(emp)
        display_result(emp, result)
        emp["name"] = name   # restore


def phase_report():
    """Print a summary of saved artifacts."""
    print("\n📋 PHASE 5 — Saved Artifacts Summary")
    print("─"*50)
    directories = ["data", "models", "outputs", "images"]
    for d in directories:
        if os.path.isdir(d):
            files = os.listdir(d)
            print(f"  📂 {d}/")
            for f in files:
                fpath = os.path.join(d, f)
                size  = os.path.getsize(fpath)
                print(f"       {f}  ({size:,} bytes)")
        else:
            print(f"  📂 {d}/  (empty)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        description="Employee Performance Predictor Pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "eda", "train", "predict", "all"],
        default="all",
        help="Which pipeline phase to run (default: all)"
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.phase in ("generate", "all"):
        phase_generate()

    if args.phase in ("eda", "all"):
        phase_eda()

    if args.phase in ("train", "all"):
        phase_train()

    if args.phase in ("predict", "all"):
        phase_predict()

    if args.phase == "all":
        phase_report()

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline finished in {elapsed:.1f} seconds.")
    print("   Check images/ for charts, models/ for saved model, outputs/ for predictions.\n")


if __name__ == "__main__":
    main()
