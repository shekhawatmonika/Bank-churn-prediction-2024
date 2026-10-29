
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Jaipur Churn Prediction - Production API")

# Load model
with open("model.pkl", "rb") as f:
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
