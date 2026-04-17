"""
eda.py
------
Exploratory Data Analysis (EDA) Module
Generates all charts and statistical insights for the HR dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("images", exist_ok=True)

LABEL_COLORS = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#27ae60"}


# ─────────────────────────────────────────────
# 1. BASIC STATS SUMMARY
# ─────────────────────────────────────────────
def print_summary(df):
    print("\n" + "="*60)
    print("  📊  DATASET SUMMARY")
    print("="*60)
    print(f"  Rows       : {df.shape[0]}")
    print(f"  Columns    : {df.shape[1]}")
    print(f"\n  Performance Distribution:")
    vc = df["performance_label"].value_counts()
    for label, count in vc.items():
        pct = count / len(df) * 100
        print(f"    {label:8s}: {count:4d}  ({pct:.1f}%)")
    print("\n  Numeric Summary:")
    print(df[["age","salary","years_experience","training_hours",
              "performance_score"]].describe().round(2).to_string())
    print("="*60)


# ─────────────────────────────────────────────
# 2. PERFORMANCE LABEL DISTRIBUTION (Bar)
# ─────────────────────────────────────────────
def plot_label_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count bar
    counts = df["performance_label"].value_counts().reindex(["Low","Medium","High"])
    colors = [LABEL_COLORS[l] for l in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Employee Performance Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Employees")
    for i, (idx, val) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(i, val + 5, str(val), ha="center", fontsize=11, fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Performance Label Share", fontsize=13, fontweight="bold")

    plt.suptitle("Target Variable Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("images/01_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved → images/01_label_distribution.png")


# ─────────────────────────────────────────────
# 3. CORRELATION HEATMAP
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).drop(
        columns=["employee_id","target"], errors="ignore"
    )
    corr = numeric_cols.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 8},
                vmin=-1, vmax=1, center=0)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved → images/02_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 4. PERFORMANCE SCORE DISTRIBUTION BY LABEL
# ─────────────────────────────────────────────
def plot_score_distributions(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in ["Low", "Medium", "High"]:
        subset = df[df["performance_label"] == label]["performance_score"]
        subset.plot.kde(ax=ax, label=label, color=LABEL_COLORS[label], linewidth=2.5)
    ax.set_xlabel("Performance Score")
    ax.set_title("Performance Score Distribution by Class", fontsize=13, fontweight="bold")
    ax.legend(title="Performance Label")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.6, label="Score = 50")
    ax.axvline(75, color="black", linestyle="--", alpha=0.6, label="Score = 75")
    plt.tight_layout()
    plt.savefig("images/03_score_distribution.png", dpi=150)
    plt.close()
    print("✅ Saved → images/03_score_distribution.png")


# ─────────────────────────────────────────────
# 5. DEPARTMENT vs PERFORMANCE (Grouped Bar)
# ─────────────────────────────────────────────
def plot_dept_performance(df):
    dept_perf = df.groupby(["department", "performance_label"]).size().unstack(fill_value=0)
    dept_perf = dept_perf.reindex(columns=["Low","Medium","High"])

    dept_perf.plot(kind="bar", figsize=(12, 6),
                   color=[LABEL_COLORS[c] for c in dept_perf.columns],
                   edgecolor="white", linewidth=0.8)
    plt.title("Performance Distribution Across Departments", fontsize=13, fontweight="bold")
    plt.xlabel("Department")
    plt.ylabel("Number of Employees")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Performance")
    plt.tight_layout()
    plt.savefig("images/04_dept_performance.png", dpi=150)
    plt.close()
    print("✅ Saved → images/04_dept_performance.png")


# ─────────────────────────────────────────────
# 6. TRAINING HOURS vs PERFORMANCE (Box Plot)
# ─────────────────────────────────────────────
def plot_training_vs_performance(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Box: training hours
    order = ["Low", "Medium", "High"]
    palette = [LABEL_COLORS[l] for l in order]
    sns.boxplot(data=df, x="performance_label", y="training_hours",
                order=order, palette=palette, ax=axes[0])
    axes[0].set_title("Training Hours by Performance Level", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Performance Label")
    axes[0].set_ylabel("Training Hours")

    # Box: absenteeism
    sns.boxplot(data=df, x="performance_label", y="absenteeism_days",
                order=order, palette=palette, ax=axes[1])
    axes[1].set_title("Absenteeism Days by Performance Level", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Performance Label")
    axes[1].set_ylabel("Absenteeism Days")

    plt.suptitle("HR Factors vs Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/05_training_absenteeism_boxplots.png", dpi=150)
    plt.close()
    print("✅ Saved → images/05_training_absenteeism_boxplots.png")


# ─────────────────────────────────────────────
# 7. SALARY vs EXPERIENCE (Scatter, coloured by label)
# ─────────────────────────────────────────────
def plot_salary_vs_experience(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in ["Low", "Medium", "High"]:
        subset = df[df["performance_label"] == label]
        ax.scatter(subset["years_experience"], subset["salary"],
                   alpha=0.45, label=label, color=LABEL_COLORS[label], s=30)
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Annual Salary (USD)")
    ax.set_title("Salary vs Experience — Coloured by Performance", fontsize=13, fontweight="bold")
    ax.legend(title="Performance")
    plt.tight_layout()
    plt.savefig("images/06_salary_vs_experience.png", dpi=150)
    plt.close()
    print("✅ Saved → images/06_salary_vs_experience.png")


# ─────────────────────────────────────────────
# 8. MANAGER RATING vs PERFORMANCE (Violin)
# ─────────────────────────────────────────────
def plot_manager_rating(df):
    order = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df, x="performance_label", y="manager_rating",
                   order=order, palette=[LABEL_COLORS[l] for l in order],
                   inner="quartile", ax=ax)
    ax.set_title("Manager Rating Distribution by Performance Level",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Performance Label")
    ax.set_ylabel("Manager Rating (1–5)")
    plt.tight_layout()
    plt.savefig("images/07_manager_rating_violin.png", dpi=150)
    plt.close()
    print("✅ Saved → images/07_manager_rating_violin.png")


# ─────────────────────────────────────────────
# 9. PROJECTS COMPLETED HISTOGRAM
# ─────────────────────────────────────────────
def plot_projects_histogram(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in ["Low", "Medium", "High"]:
        subset = df[df["performance_label"] == label]["projects_completed"]
        ax.hist(subset, bins=15, alpha=0.6, label=label,
                color=LABEL_COLORS[label], edgecolor="white")
    ax.set_xlabel("Projects Completed")
    ax.set_ylabel("Count")
    ax.set_title("Projects Completed by Performance Label", fontsize=13, fontweight="bold")
    ax.legend(title="Performance")
    plt.tight_layout()
    plt.savefig("images/08_projects_histogram.png", dpi=150)
    plt.close()
    print("✅ Saved → images/08_projects_histogram.png")


# ─────────────────────────────────────────────
# RUN ALL EDA
# ─────────────────────────────────────────────
def run_eda(df):
    print("\n📊 Running Exploratory Data Analysis …")
    print_summary(df)
    plot_label_distribution(df)
    plot_correlation_heatmap(df)
    plot_score_distributions(df)
    plot_dept_performance(df)
    plot_training_vs_performance(df)
    plot_salary_vs_experience(df)
    plot_manager_rating(df)
    plot_projects_histogram(df)
    print("\n✅ EDA complete — all images saved in images/")


if __name__ == "__main__":
    from src.preprocess import load_data, clean_data, engineer_features, encode_features
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_features(df)
    run_eda(df)
