
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Jaipur Churn Prediction - Production API")

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# EXACT MAPPING of Geography
GEOGRAPHY_CODES = {"Jaipur City": 0, "Jaipur Gramin": 1, "Jaipur Urban": 2}

class ChurnInput(BaseModel):
    CreditScore: int
    Geography: str  # Jaipur City / Jaipur Gramin / Jaipur Urban
    Gender: str     # Male / Female
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def root():
    return {"message": "Churn API is LIVE - Jaipur Model", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    try:
        # Convert string to number 
        geo_code = GEOGRAPHY_CODES.get(data.Geography, 0)
        
        # Gender: Female=0, Male=1
        if isinstance(data.Gender, str):
            gender_code = 0 if data.Gender.lower() == "female" else 1
        else:
            gender_code = int(data.Gender)

        # 
        columns = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
        
        row = [[data.CreditScore, geo_code, gender_code, data.Age, data.Tenure, data.Balance, data.NumOfProducts, data.HasCrCard, data.IsActiveMember, data.EstimatedSalary]]
        
        df = pd.DataFrame(row, columns=columns)

        
        if hasattr(model, 'feature_names_in_'):
            df = df[model.feature_names_in_]

        if scaler:
            df = scaler.transform(df)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else float(pred)

        return {
            "churn_prediction": int(pred),
            "churn_probability": float(prob),
            "message": "Customer will Churn" if pred == 1 else "Customer will NOT Churn",
            "input_converted": {"Geography_code": geo_code, "Gender_code": gender_code}
        }
    except Exception as e:
        return {"error": str(e)}
=======
import streamlit as st
import pickle
import pandas as pd

# Model aur Scaler load
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


st.title("🏦 Bank Churn Prediction")
st.subheader("Jaipur City | Gramin | Urban")

credit = st.number_input("Credit Score", 300, 900, 650)

# Exact Encoding mapping :
# Jaipur City: 0, Jaipur Gramin: 1, Jaipur Urban: 2
geography_codes = {"Jaipur City": 0, "Jaipur Gramin": 1, "Jaipur Urban": 2}
geography_name = st.selectbox("Geography", list(geography_codes.keys()))

gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 21326337.65, 50000.0)
products = st.selectbox("Num Of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 16999360.8, 90000.0)

if st.button("🔮 Predict", use_container_width=True):
    # Sahi order aur column names
    input_data = pd.DataFrame(
        [[credit, geography_codes[geography_name], gender, age, tenure, balance, products, has_card, is_active, salary]],
        columns=[
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]
    )
    
    # Scale inputs
    data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
    
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1] * 100

    if pred == 1:
        st.error(f"⚠️ Churn Hoga - {prob:.1f}% Risk")
        st.write("Jaipur Urban/Gramin me retention offer do")
    else:
        st.success(f"✅ Rahega - Safe: {100 - prob:.1f}%")

