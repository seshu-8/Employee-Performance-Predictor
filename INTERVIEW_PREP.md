# 🎓 Interview Preparation Guide
## Employee Performance Predictor — HR & Technical Questions

---

## 🗣️ HOW TO EXPLAIN TO AN HR INTERVIEWER (Non-Technical)

> "I built a system that analyses employee data — things like training hours,
> absenteeism, manager ratings, and project delivery — and predicts whether
> an employee is a High, Medium, or Low performer. The system can help HR
> teams make more consistent, data-backed decisions for promotions, training
> plans, and retention strategies. Instead of relying purely on gut feeling,
> HR can now get an AI-generated recommendation with actionable next steps."

---

## 💻 HOW TO EXPLAIN TO A TECHNICAL INTERVIEWER

> "This is an end-to-end classification pipeline. I generated a synthetic HR
> dataset with 1,000 records and 15 features using domain-realistic formulas.
> After cleaning and feature engineering — adding derived KPIs like engagement
> score and productivity index — I trained three classifiers: Logistic
> Regression, Random Forest, and Gradient Boosting. Random Forest achieved
> the best test accuracy of ~88% with 5-fold stratified cross-validation. I
> saved the model with joblib and built a prediction engine that outputs
> per-class probabilities and HR-specific action items."

---

## 📋 10 INTERVIEW QUESTIONS + STRONG ANSWERS

---

### Q1. Why did you choose Random Forest as your final model?

**A:** Random Forest consistently outperformed the other models because:
- It handles a mix of continuous and encoded categorical features natively
- It's robust to outliers in features like salary and overtime hours
- It provides feature importance scores, which are directly useful for HR storytelling
- It avoids overfitting through bagging (averaging many decision trees)
- Cross-validation accuracy was stable (~86%), showing it generalises well

---

### Q2. How did you handle class imbalance in the dataset?

**A:** The dataset had a naturally skewed distribution (more Medium performers than Low/High). I handled this by:
1. Using `stratify=y` in `train_test_split` — ensures each split has proportional class representation
2. Evaluating with per-class precision, recall, and F1-score, not just overall accuracy
3. Reporting weighted F1 score alongside accuracy for a fairer evaluation

---

### Q3. What is feature engineering and what features did you create?

**A:** Feature engineering is the process of creating new, more informative input variables from existing data. I created:
- `engagement_score` = average of manager rating + employee satisfaction — a composite HR KPI
- `productivity_index` = projects completed / (experience + 1) — output normalised for experience
- `salary_per_exp_year` = salary / (experience + 1) — measures compensation alignment
- `overwork_flag` = binary flag if overtime > 20 hrs — captures burnout risk
- `high_absenteeism` = binary flag if absent > 10 days — flags disengagement signal

---

### Q4. What is a confusion matrix? How do you read it?

**A:** A confusion matrix is a table showing how many predictions were correct vs incorrect for each class. For a 3-class problem (Low/Medium/High):
- **Diagonal values** = correct predictions
- **Off-diagonal values** = misclassifications
- If the model confuses Low with Medium often, the [Low, Medium] cell will be high
- It helps identify which class the model struggles with most, guiding further tuning

---

### Q5. What is cross-validation and why did you use it?

**A:** Cross-validation (k-fold) splits the training data into k subsets (I used 5). The model is trained on k-1 folds and evaluated on the remaining fold, k times. This gives a more reliable estimate of generalisation performance than a single train-test split — especially important with 1,000 rows where a single split could be lucky or unlucky.

---

### Q6. How did you create the dataset without real company data?

**A:** I built a synthetic dataset using domain knowledge and realistic HR distributions:
- Features like salary were tied to job level with realistic base rates
- Performance score was computed using a weighted formula: training hours, absenteeism, projects, manager rating, etc.
- Gaussian noise was added to prevent the model from learning perfectly artificial patterns
- The result is a statistically realistic proxy for real HR data

---

### Q7. How would you deploy this system in a real company?

**A:** The deployment path would be:
1. **API Layer** — Wrap the predict.py engine in a FastAPI REST endpoint
2. **Frontend** — Build a Streamlit or React UI for HR managers to input employee data
3. **Integration** — Connect to the HRMS (e.g., SAP SuccessFactors, Workday) via API
4. **Monitoring** — Track prediction drift over time and retrain as new data comes in
5. **Explainability** — Add SHAP values so HR can see *why* a prediction was made

---

### Q8. What are the ethical concerns with this kind of AI system?

**A:** Important ethical considerations include:
- **Bias** — If historical HR data reflects biases (e.g., gender pay gaps), the model will learn them
- **Transparency** — Employees should know they're being evaluated by AI and understand why
- **Fairness** — The model should be audited across demographic groups for disparate impact
- **Human oversight** — AI should assist HR decisions, not replace human judgment
- **Data privacy** — Employee data must be anonymised, secured, and GDPR-compliant

---

### Q9. What does the StandardScaler do? Why is it necessary?

**A:** StandardScaler normalises each feature to have mean=0 and standard deviation=1. It's necessary because:
- Salary (range: 30k–200k) would dominate age (22–60) without scaling
- Distance-sensitive models like Logistic Regression are particularly affected
- Gradient Boosting and Random Forest are tree-based and less affected, but scaling ensures consistency across models
- The scaler is fitted *only* on training data and applied to test data — this prevents data leakage

---

### Q10. What would you improve if you had more time?

**A:** Several strong improvements:
1. **Real dataset** — IBM HR Analytics on Kaggle for real-world validation
2. **SHAP explainability** — `shap.TreeExplainer` to explain individual predictions
3. **Streamlit dashboard** — Interactive UI with sliders and real-time predictions
4. **Hyperparameter tuning** — `GridSearchCV` or `Optuna` to optimise Random Forest
5. **Employee attrition model** — Predict who will leave (churn prediction)
6. **Time-series analysis** — Track performance trajectory over multiple review cycles

---

## 🌟 BONUS: Project Metrics to Mention

| Metric | Value |
|---|---|
| Dataset size | 1,000 employees |
| Features | 15 raw + 6 engineered = 21 total |
| Models trained | 3 |
| Best accuracy | ~88% (Random Forest) |
| Cross-validation | 5-fold stratified |
| Charts generated | 10 EDA + 2 model charts |
| HR use cases covered | 5 (promotion, training, retention, review, planning) |
