import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

def load_model():
    best_model = joblib.load('models/best_performing_model.joblib')
    preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.joblib')
    return best_model, preprocessing_pipeline

def run_streamlit_app():
    """Run the Streamlit web application for loan prediction."""
    best_model, preprocessing_pipeline = load_model()

    st.set_page_config(page_title="Loan Prediction App", page_icon=":money_with_wings:")
    st.title("Loan Prediction App")

    # Input features
    loan_amount = st.number_input("Loan Amount", min_value=5000, max_value=100000, step=1000)
    interest_rate = st.number_input("Interest Rate", min_value=0.05, max_value=0.25, step=0.01)
    term = st.number_input("Term (months)", min_value=12, max_value=60, step=1)
    credit_score = st.number_input("Credit Score", min_value=500, max_value=850, step=1)
    annual_income = st.number_input("Annual Income", min_value=20000, max_value=150000, step=1000)
    employment_status = st.selectbox("Employment Status", options=['Employed', 'Self-employed', 'Unemployed'])
    home_ownership = st.selectbox("Home Ownership", options=['Own', 'Rent', 'Mortgage'])
    purpose = st.selectbox("Purpose", options=['Debt consolidation', 'Home improvement', 'Business', 'Personal'])
    state = st.text_input("State")
    preferred_loan_date = st.date_input("Preferred Loan Date")

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'loan_amount': [loan_amount],
        'interest_rate': [interest_rate],
        'term': [term],
        'credit_score': [credit_score],
        'annual_income': [annual_income],
        'employment_status': [employment_status],
        'home_ownership': [home_ownership],
        'purpose': [purpose],
        'state': [state],
        'preferred_loan_date': [preferred_loan_date]
    })

    # Preprocess the input data
    input_data_processed = preprocessing_pipeline.transform(input_data)

    # Make prediction
    if st.button("Predict"):
        prediction = best_model.predict(input_data_processed)
        if prediction[0] == 'Approved':
            st.success("The loan is approved!")
        else:
            st.error("The loan is rejected.")

if __name__ == "__main__":
    run_streamlit_app()
