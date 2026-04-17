# 🧠 Employee Performance Predictor
### *Data Analytics + Machine Learning for HR Decision-Making*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## 📌 Project Overview

This end-to-end ML project predicts **employee performance levels** (High / Medium / Low) using synthetic HR data and machine learning. It simulates how real companies use people analytics to make data-driven HR decisions.

> Built as a **student placement portfolio project** — no real company data required.

---

## 🎯 Problem Statement

HR teams in large organisations struggle to proactively identify:
- Employees at risk of low performance before it impacts the business
- High performers eligible for promotion or leadership roles
- Employees who need targeted training and development

**This system automates that identification using machine learning.**

---

## 💼 Business Value

| HR Use Case | How This Project Helps |
|---|---|
| **Performance Reviews** | Objective ML-based scoring alongside manager ratings |
| **Promotion Decisions** | Data-driven identification of top performers |
| **Training Planning** | Spot low performers who need upskilling |
| **Retention Strategy** | Flag at-risk employees before they disengage |
| **Workforce Planning** | Department-level performance analytics |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression) |
| **Visualisation** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Notebook** | Jupyter |

---

## 🏗️ Architecture

```
Raw HR Data (Synthetic)
        │
        ▼
┌───────────────────┐
│  Data Generation  │  → 1000 employees, 15+ features
│  (generate_data)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Preprocessing    │  → Clean, Encode, Scale
│  (preprocess.py)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Feature Eng.     │  → Derived KPIs (engagement score, productivity index…)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Training   │  → 3 models → Best selected automatically
│  (train_model.py) │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Prediction &     │  → Single / Batch prediction + HR insights
│  Insights         │
└────────┬──────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  HR Decision Dashboard   │
  │  Charts + Reports        │
  └──────────────────────────┘
```

---

## 📂 Folder Structure

```
Employee-Performance-Predictor/
│
├── data/
│   └── employee_data.csv         ← Auto-generated synthetic dataset
│
├── notebooks/
│   └── 01_EDA_and_Model.ipynb   ← Full interactive analysis
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py          ← Synthetic HR data generator
│   ├── preprocess.py             ← Cleaning + feature engineering
│   ├── eda.py                    ← EDA charts and stats
│   ├── train_model.py            ← Model training + evaluation
│   └── predict.py                ← Prediction engine + HR insights
│
├── models/
│   ├── best_model.pkl            ← Saved trained model
│   ├── scaler.pkl                ← Feature scaler
│   └── label_encoders.pkl        ← Categorical encoders
│
├── outputs/
│   ├── predictions.csv           ← Batch prediction results
│   └── model_results.json        ← Accuracy comparison JSON
│
├── images/
│   ├── 01_label_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_score_distribution.png
│   ├── 04_dept_performance.png
│   ├── 05_training_absenteeism_boxplots.png
│   ├── 06_salary_vs_experience.png
│   ├── 07_manager_rating_violin.png
│   ├── 08_projects_histogram.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── main.py                       ← 🚀 One-command pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor
```

### Step 2 — Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Full Pipeline (One Command)
```bash
python main.py
```

### Option B — Individual Phases
```bash
python main.py --phase generate    # Generate synthetic dataset
python main.py --phase eda         # Run Exploratory Data Analysis
python main.py --phase train       # Train and evaluate models
python main.py --phase predict     # Run sample predictions
```

### Option C — Jupyter Notebook
```bash
jupyter notebook notebooks/01_EDA_and_Model.ipynb
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `age` | Employee age (22–60) |
| `gender` | Male / Female / Non-Binary |
| `education_level` | High School / Bachelor's / Master's / PhD |
| `department` | Engineering, Sales, HR, Finance, Marketing, Operations, Support |
| `job_level` | Junior / Mid / Senior / Lead / Manager |
| `years_experience` | Total work experience |
| `salary` | Annual compensation (USD) |
| `training_hours` | Training hours completed |
| `absenteeism_days` | Days absent per year |
| `projects_completed` | Number of projects delivered |
| `overtime_hours` | Monthly overtime hours |
| `manager_rating` | Manager's rating (1.0–5.0) |
| `employee_satisfaction` | Self-reported satisfaction (1.0–5.0) |
| `years_since_promotion` | Years since last promotion |
| `performance_score` | Composite score (0–100) |
| `performance_label` | **Target: Low / Medium / High** |

---

## 🤖 Model Results

| Model | Test Accuracy | CV Accuracy |
|---|---|---|
| Logistic Regression | ~78% | ~76% |
| **Random Forest** | **~88%** | **~86%** |
| Gradient Boosting | ~87% | ~85% |

> Best model saved automatically to `models/best_model.pkl`

---

## 📈 Key Insights Discovered

1. **Training hours** — High performers average 2× more training than Low performers
2. **Absenteeism** — Low performers have 3× higher absenteeism rate
3. **Manager Rating** — Single strongest predictor of performance tier
4. **Projects Completed** — High correlates with High performance (obvious but confirmed)
5. **Overwork Risk** — Excessive overtime (>20h/month) is associated with Medium, not High performance

---

## 🔮 Sample Prediction Output

```
╔══════════════════════════════════════════════════════════╗
  🏢  EMPLOYEE PERFORMANCE PREDICTION REPORT
════════════════════════════════════════════════════════════
  Employee ID   : 9001
  Department    : Engineering
  Job Level     : Senior
  Experience    : 8 years
────────────────────────────────────────────────────────────
  ⚡ Prediction  : 🟢  High Performer

  📊 Confidence Scores:
     Low     :   3.2%  ▌
     Medium  :  11.5%  ██
     High    :  85.3%  █████████████████

  💼 HR Action Items:
     • 🏆 Promote to leadership or senior role — top performer.
     • 💡 Assign mentoring responsibilities to leverage expertise.
     • 📈 Consider fast-track career development programme.
     • 💰 Review compensation to ensure market alignment.
```

---

## 🔄 Future Improvements

- [ ] **Streamlit Dashboard** — Interactive web app for HR teams
- [ ] **Employee Attrition Prediction** — Predict who will leave
- [ ] **Real-time API** — FastAPI endpoint for live HR systems
- [ ] **Deep Learning** — Neural network for complex pattern detection
- [ ] **Real Dataset** — IBM HR Analytics or Kaggle HR dataset
- [ ] **SHAP Explainability** — Why did the model predict this score?

---

## 🎓 Interview Preparation

### Q: What problem does this project solve?
**A:** It automates employee performance classification using ML, helping HR teams make objective, data-driven decisions on promotions, training, and retention — reducing bias in performance reviews.

### Q: Why Random Forest?
**A:** It handles mixed feature types well, is robust to outliers, provides feature importances, and consistently outperforms simpler models on tabular HR data.

### Q: How did you handle class imbalance?
**A:** Used `stratify=y` in train-test split to maintain proportional class distribution, and evaluated with per-class precision/recall, not just accuracy.

---

## 👨‍💻 Author

**[Your Name]**  
B.Tech [Your Branch] | [Your College]  
📧 your.email@example.com  
🔗 [LinkedIn](https://www.linkedin.com/in/seshu-babu-konijeti-74968b2b9?utm_source=share_via&utm_content=profile&utm_medium=member_android) | [GitHub]([https://github.com](https://github.com/seshu-8)

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.
