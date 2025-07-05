import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Load model and scaler
model = joblib.load("model/credit_risk_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("💳 Credit Risk Evaluator")
st.markdown("Predict if a customer will default on their credit payment next month.")
st.markdown("""
### 📘 Instructions

This tool evaluates the credit risk of a customer based on historical payment behavior and demographic data.

**Please enter the following details:**

- `LIMIT_BAL`: Total credit limit assigned to the user (in ₹)
- `SEX`: Gender (1 = Male, 2 = Female)
- `EDUCATION`: Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)
- `MARRIAGE`: Marital status (1 = Married, 2 = Single, 3 = Others)
- `AGE`: Age of the applicant in years

**Repayment Status:**
- `PAY_0` to `PAY_6`: Past monthly payment status (e.g., 0 = on time, 1 = 1 month delay, -1 = early payment)

**Billing Amounts:**
- `BILL_AMT1` to `BILL_AMT6`: Monthly bill statements for the past 6 months

**Payment Amounts:**
- `PAY_AMT1` to `PAY_AMT6`: Amount actually paid in the last 6 months

---

👉 Once you’ve entered all fields, click **Predict Credit Risk** to view the result and SHAP-based explanation.
""")


# Input fields
LIMIT_BAL = st.number_input("Credit Limit (₹)", min_value=10000, max_value=1000000, step=10000)
SEX = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
EDUCATION = st.selectbox("Education Level", [1, 2, 3, 4], format_func=lambda x: {
    1: "Graduate School", 2: "University", 3: "High School", 4: "Others"
}[x])
MARRIAGE = st.selectbox("Marital Status", [1, 2, 3], format_func=lambda x: {
    1: "Married", 2: "Single", 3: "Other"
}[x])
AGE = st.slider("Age", 18, 75, 30)

# Past payment behavior
PAY = [st.selectbox(f"Repayment Status {i} Months Ago", list(range(-2, 9)), key=f"PAY_{i}") for i in range(0, 6)]

# Past bill statements
BILL = [st.number_input(f"BILL_AMT{i+1}", value=0) for i in range(6)]

# Past payments
PAY_AMT = [st.number_input(f"PAY_AMT{i+1}", value=0) for i in range(6)]

# When Predict button is clicked
if st.button("Predict Credit Risk"):
    # Combine all inputs
    input_data = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE] + PAY + BILL + PAY_AMT
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # SHAP for individual explanation
    st.subheader("📉 SHAP Explanation for This Prediction")

# Prepare explainer
    explainer = shap.Explainer(model, feature_names=[
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ])

    shap_value = explainer(input_scaled)

    # Render SHAP force plot
    shap_html = shap.plots.force(shap_value[0], matplotlib=False, show=False)
    components.html(shap.getjs() + shap_html.html(), height=300)


    if prediction == 1:
        st.error(f"🚫 High Risk of Default (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {probability:.2f})")
