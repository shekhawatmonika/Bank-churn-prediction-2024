🏦 Bank Customer Churn Prediction - Jaipur 2024 | Live Project
# Bank Customer Churn Prediction - Jaipur 2024 | Live Project

🔗 **Live Demo:** https://bank-churn-prediction-2024.streamlit.app/
🔗 **API (Render):** https://bank-churn-prediction-2024.onrender.com/docs

> An end-to-end Machine Learning solution to predict bank customer churn for Jaipur region - with focus on **no overfitting and production-ready deployment**.

### 📌 Problem Statement
Banks in Jaipur are losing ~20-25% customers annually. Retaining an existing customer is 5x cheaper than acquiring a new one.

**Business Goal:** Identify customers who are likely to churn *before* they leave, so the retention team can take proactive action.

### 📂 Dataset: `churn_jaipur_updated.csv`
Localized Jaipur banking data with key features: Geography (Jaipur City/Gramin/Urban), Gender, Age, Balance, etc.

### 🚀 Tech Stack
- Model: RandomForest / XGBoost with `model.pkl` & `scaler.pkl`
- Backend: FastAPI + Docker (Deployed on Render)
- Frontend: Streamlit (Deployed on Streamlit Cloud)

### ▶️ How to Run Locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload
streamlit run Streamlit_app.py