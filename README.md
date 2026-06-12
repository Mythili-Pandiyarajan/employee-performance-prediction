# 👔 Employee Performance Prediction — INX Future Inc.

An end-to-end Machine Learning project to predict employee performance ratings and identify key factors driving performance across departments.

## 🚀 Live Demo
🌐 **Streamlit:** https://employee-performance-prediction-mythili.streamlit.app/

---

## 🔍 Project Overview

INX Future Inc. is facing performance challenges across departments. This project analyses **1200 employee records (28 features)** to identify what drives performance, build a predictive model, and provide actionable HR recommendations.

**4 Key Deliverables:**
1. Department-wise performance analysis
2. Top 3 performance-influencing factors
3. Trained prediction model for deployment
4. Recommendations to improve employee performance

**3 Performance Classes:**
- Rating 2 — Low Performer
- Rating 3 — Good Performer *(majority class — 72.8%)*
- Rating 4 — Excellent Performer

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `INX_Future_Employee_Performance.ipynb` | End-to-end ML notebook — EDA, models, SHAP, recommendations |
| `inx_employee_performance_model.pkl` | Best model — Random Forest Baseline |
| `inx_feature_columns.pkl` | Feature column order for inference |
| `inx_label_encoders.pkl` | Label encoders for categorical features |

---

## 📊 Model Performance

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| **Random Forest (Baseline) ✅** | **0.9375** | **0.8983** |
| Gradient Boosting (Baseline) | 0.9250 | 0.8845 |
| Decision Tree (Baseline) | 0.8792 | 0.7965 |
| Logistic Regression (Baseline) | 0.8250 | 0.7321 |
| Random Forest (Tuned) | 0.9292 | 0.8852 |

✅ **Best Model: Random Forest Baseline** — Accuracy **93.75%**, F1 Macro **0.8983**
- Tuned model performed slightly lower — default depth was already optimal for this dataset size
- Gradient Boosting is competitive but Random Forest generalizes better

---

## 🏆 Top 3 Performance Factors (Deliverable 2)

| Rank | Feature | Insight |
|------|---------|---------|
| 1 | `EmpLastSalaryHikePercent` | Higher salary hikes strongly predict higher performance — recognition drives motivation |
| 2 | `EmpEnvironmentSatisfaction` | Positive workplace environment directly lifts performance ratings |
| 3 | `YearsSinceLastPromotion` | Longer stagnation without promotion correlates with declining performance |

Confirmed by both **Random Forest Feature Importance** and **SHAP TreeExplainer**.

---

## 🏢 Department-wise Performance (Deliverable 1)

| Department | Avg Rating | Status |
|-----------|------------|--------|
| Development | 3.086 | 🟢 Highest |
| Data Science | 3.050 | 🟢 High |
| Research & Development | ~3.0 | 🟡 Moderate |
| Human Resources | 2.926 | 🟡 Moderate |
| Sales | 2.861 | 🔴 Low |
| Finance | 2.776 | 🔴 Lowest |

---

## 📁 Dataset

- **1200 employee records**, 28 features
- **Numerical features:** Age, salary hike %, work experience, satisfaction scores, training times
- **Categorical features:** Department, job role, gender, marital status, overtime, attrition
- **Target:** `PerformanceRating` — values 2, 3, 4 (multiclass, imbalanced)
- **No missing values, no duplicates**
- Source: INX Future Inc. HR Dataset (Project Code: 10281)

---

## 📓 Notebook Structure

1. Import Libraries & Load Data
2. Data Overview
3. Missing Value & Duplicate Check
4. Target Variable Distribution
5. Univariate Analysis
6. Department-wise Performance Analysis *(Deliverable 1)*
7. Bivariate Analysis
8. Skewness Check
9. Correlation Analysis
10. Label Encoding
11. Feature Selection — Top 3 Factors *(Deliverable 2)*
12. Train-Test Split
13. Baseline Models (LR, DT, RF, GBM)
14. Overfitting Check
15. Hyperparameter Tuning (GridSearchCV)
16. Cross Validation
17. SHAP Explainability
18. Model Comparison Report
19. Final Model & Prediction System *(Deliverable 3)*
20. Recommendations to Improve Performance *(Deliverable 4)*

---

## 🛠️ Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP
- Joblib

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/Mythili-Pandiyarajan/inx-employee-performance.git
cd inx-employee-performance
pip install -r requirements.txt
jupyter notebook INX_Future_Employee_Performance.ipynb
```

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)

- **ML:** Random Forest, Gradient Boosting, Decision Tree, Logistic Regression
- **Explainability:** SHAP TreeExplainer
- **Deployment:** joblib model serialization

---

## 💡 Key Findings

- **Salary hike is the strongest predictor** — merit-based compensation directly drives performance
- **Random Forest Baseline beats all models** including the tuned variant — dataset size (1200 rows) means default depth is already optimal
- **Development and Data Science departments** outperform Sales and Finance — their practices should be studied and replicated
- **SHAP confirms feature importance** — top 3 features from both methods are identical, validating the model's learned patterns
- **Class imbalance (72.8% Rating 3)** handled via F1 Macro as primary metric — avoids misleading accuracy

---

## 💼 Recommendations (Deliverable 4)

1. **Reward high performers with competitive salary hikes** — merit-based review cycles, not flat hike structures
2. **Invest in work environment and culture** — physical workspace, psychological safety, teamwork
3. **Establish timely promotion pathways** — structured criteria, transparent timelines, regular career conversations
4. **Address department-specific gaps** — targeted L&D for Sales, Finance, and HR departments
5. **Improve job involvement and role clarity** — flexible policies, manageable workloads
6. **Integrate the predictive model into hiring** — screen candidates by predicted performance rating during recruitment

---

## 👩‍💻 Author

**Mythili Pandiyarajan** — [GitHub Profile](https://github.com/Mythili-Pandiyarajan)
