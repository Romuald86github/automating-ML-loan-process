# src/app/app.py

import streamlit as st
import pandas as pd
import joblib

def load_model():
    best_model = joblib.load('models/best_performing_model.joblib')
    preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.joblib')
    selected_feature_names = joblib.load('models/selected_feature_names.joblib')
    return best_model, preprocessing_pipeline, selected_feature_names

def run_streamlit_app():
    """Run the Streamlit web application for loan prediction."""
    best_model, preprocessing_pipeline, selected_feature_names = load_model()

    st.set_page_config(page_title="Loan Prediction App", page_icon=":money_with_wings:")
    st.title("Loan Prediction App")

    # Input features
    input_features = {}
    for feature in selected_feature_names:
        if feature == 'loan_amount':
            input_features[feature] = st.number_input("Loan Amount", min_value=5000, max_value=100000, step=1000)
        elif feature == 'interest_rate':
            input_features[feature] = st.number_input("Interest Rate", min_value=0.05, max_value=0.25, step=0.01)
        elif feature == 'term':
            input_features[feature] = st.number_input("Term (months)", min_value=12, max_value=60, step=1)
        elif feature == 'credit_score':
            input_features[feature] = st.number_input("Credit Score", min_value=500, max_value=850, step=1)
        elif feature == 'annual_income':
            input_features[feature] = st.number_input("Annual Income", min_value=20000, max_value=150000, step=1000)
        elif feature == 'employment_status':
            input_features[feature] = st.selectbox("Employment Status", options=['Employed', 'Self-employed', 'Unemployed'])
        elif feature == 'home_ownership':
            input_features[feature] = st.selectbox("Home Ownership", options=['Own', 'Rent', 'Mortgage'])
        elif feature == 'purpose':
            input_features[feature] = st.selectbox("Purpose", options=['Debt consolidation', 'Home improvement', 'Business', 'Personal'])
        elif feature == 'state':
            input_features[feature] = st.text_input("State")
        elif feature == 'preferred_loan_date':
            input_features[feature] = st.date_input("Preferred Loan Date")

    # Convert the input features into a DataFrame
    input_data = pd.DataFrame(input_features, index=[0])

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
