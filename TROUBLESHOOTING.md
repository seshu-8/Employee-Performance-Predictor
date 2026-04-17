# 🔧 Troubleshooting Guide

## Common Errors & Solutions

---

### ❌ Error: `ModuleNotFoundError: No module named 'sklearn'`
**Cause:** scikit-learn not installed  
**Fix:**
```bash
pip install scikit-learn
# or reinstall all:
pip install -r requirements.txt
```

---

### ❌ Error: `ModuleNotFoundError: No module named 'src'`
**Cause:** Running script from wrong directory  
**Fix:** Always run from project root:
```bash
cd Employee-Performance-Predictor
python main.py          # ✅ correct
python src/main.py      # ❌ wrong
```

---

### ❌ Error: `FileNotFoundError: data/employee_data.csv`
**Cause:** Dataset not generated yet  
**Fix:**
```bash
python main.py --phase generate
```

---

### ❌ Error: `FileNotFoundError: models/best_model.pkl`
**Cause:** Model not trained yet  
**Fix:**
```bash
python main.py --phase train
```

---

### ❌ Error: `ValueError: Unknown label type` in LabelEncoder
**Cause:** Passing an unseen category (e.g., department not in training data)  
**Fix:** Check valid categories in predict.py. Current valid values:
- `department`: Engineering, Sales, HR, Finance, Marketing, Operations, Support
- `job_level`: Junior, Mid, Senior, Lead, Manager
- `education_level`: High School, Bachelor's, Master's, PhD
- `gender`: Male, Female, Non-Binary

---

### ❌ Error: Jupyter notebook shows `Kernel not found`
**Fix:**
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv
# Then select 'venv' kernel in Jupyter
```

---

### ❌ Error: Charts not displaying in script mode
**Cause:** `plt.show()` blocks headless environments  
**Fix:** Already handled — scripts use `matplotlib.use("Agg")` and save to `images/`

---

### ❌ Error: `ConvergenceWarning` in Logistic Regression
**Cause:** Not enough iterations  
**Fix:** Already set `max_iter=1000` in train_model.py. Safe to ignore this warning.

---

### ❌ Error: `Permission denied` writing to models/ or outputs/
**Fix:**
```bash
# Mac/Linux
chmod -R 755 .
# Windows: Run terminal as Administrator
```

---

### ℹ️ Low Accuracy (<75%)
**Cause:** Small dataset or random seed variation  
**Fix:**
```bash
# Increase dataset size in generate_data.py:
df = generate_dataset(n=5000)  # try 5000 employees
```

---

### ℹ️ Slow Training
**Cause:** Large n_estimators on a slow machine  
**Fix:** Reduce estimators in train_model.py:
```python
RandomForestClassifier(n_estimators=100, ...)   # was 200
GradientBoostingClassifier(n_estimators=80, ...) # was 150
```
