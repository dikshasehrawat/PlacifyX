# 🎓 PlacifyX — AI-Powered Campus Placement Prediction

> An end-to-end Machine Learning system that predicts campus placement outcomes with Explainable AI (SHAP), bias analysis, and an interactive Streamlit web app.

---

## 🚀 Live Demo
[Coming soon after deployment]

---

## 📌 Problem Statement
Campus placements are a critical milestone for engineering students. This project predicts whether a student will get placed based on their academic profile, skills, and activities — and explains *why* using Explainable AI.

---

## 🧠 Key Features
- **Placement Prediction** — Predict placement probability based on 10 student features
- **Explainable AI (SHAP)** — Understand why the model made each prediction
- **Profile Comparison** — Compare your profile vs average placed student
- **Model Insights** — Confusion matrix, ROC curve, model comparison
- **Bias & Fairness Analysis** — Subgroup performance analysis across extracurriculars and placement training

---

## 📊 Dataset
- **Source:** [Campus Placement Dataset](https://www.kaggle.com/datasets/ruchikakumbhar/placement-prediction-dataset) by Ruchika Kumbhar (Kaggle)
- **Size:** 10,000 students, 12 features
- **Target:** PlacementStatus (Placed / NotPlaced)
- **Placement Rate:** 41.97%

---

## 🔍 Key EDA Insights
- Students with extracurricular activities have **62% placement rate** vs **13.7%** without
- Placement training increases placement rate by **36%**
- Aptitude Test Score is the **strongest predictor** (correlation: 0.52)
- CGPA gap: Placed mean **8.02** vs NotPlaced mean **7.47**
- Internships show non-linear pattern — **2 internships** needed for significant impact

---

## 🤖 Models Trained & Compared

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** ✅ | 80.85% | 77.27% | 88.37% |
| Random Forest | 78.75% | 74.10% | 86.68% |
| XGBoost | 78.20% | 73.83% | 85.78% |

> Best model selected based on ROC-AUC score

---

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **ML:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud

---

## 📁 Project Structure
PlacifyX/
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Cleaned dataset
├── notebooks/
│   └── 01_EDA.ipynb          # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py # Data cleaning & splitting
│   ├── train.py              # Model training & comparison
│   ├── evaluate.py           # Model evaluation plots
│   ├── explain.py            # SHAP global explanations
│   └── predict.py            # Single prediction + SHAP
├── models/
│   └── model.pkl             # Saved best model
├── app/
│   └── streamlit_app.py      # Streamlit web application
├── images/                   # Saved plots
├── config.py                 # Project configuration
├── requirements.txt          # Dependencies
└── README.md
---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/dikshasehrawat/PlacifyX.git
cd PlacifyX

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Run the app
streamlit run app/streamlit_app.py
```

---

## 📈 SHAP Explainability
PlacifyX uses SHAP (SHapley Additive exPlanations) to explain individual predictions:
- **Global explanation** — which features matter most overall
- **Individual explanation** — why the model predicted Placed/NotPlaced for a specific student
- **Fairness analysis** — whether the model performs equally across subgroups

---

## ⚖️ Bias & Fairness
- Checked model performance across extracurricular and placement training subgroups
- Identified real-world disparities in placement rates
- Discussed fairness implications and limitations

---

## 👩‍💻 Author
**Diksha Sehrawat**
- GitHub: https://github.com/dikshasehrawat
- LinkedIn: https://www.linkedin.com/in/diksha-sehrawat-7a8aa12a3/

---

## 📄 License
MIT License