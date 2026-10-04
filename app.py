import streamlit as st
import pickle
import pandas as pd

# Model aur Scaler load karein
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


st.title("🏦 Bank Churn Prediction")
st.subheader("Jaipur City | Gramin | Urban")

credit = st.number_input("Credit Score", 300, 900, 650)

# Exact Encoding mapping jo LabelEncoder ne banayi:
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