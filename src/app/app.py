import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_model():
    best_model = joblib.load('models/best_performing_model.joblib')
    preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.joblib')
    selected_feature_names = joblib.load('models/selected_feature_names.joblib')

    # Extract the OneHotEncoder from the ColumnTransformer
    categorical_encoder = next(step for step in preprocessing_pipeline.named_steps['categorical_encoder'].transformers_ if isinstance(step[1], OneHotEncoder))[1]

    return best_model, preprocessing_pipeline, selected_feature_names, categorical_encoder

def run_streamlit_app():
    """Run the Streamlit web application for loan prediction."""
    best_model, preprocessing_pipeline, selected_feature_names, categorical_encoder = load_model()

    st.set_page_config(page_title="Loan Prediction App", page_icon=":money_with_wings:")
    st.title("Loan Prediction App")

    # Input features
    input_data = {}
    for feature_name in selected_feature_names:
        if feature_name in ['loan_amount', 'interest_rate', 'term', 'credit_score', 'annual_income']:
            input_data[feature_name] = st.number_input(f"{feature_name.replace('_', ' ').capitalize()}", min_value=0, step=1)
        else:
            input_data[feature_name] = st.selectbox(f"{feature_name.replace('_', ' ').capitalize()}", options=categorical_encoder.categories_[selected_feature_names.index(feature_name)])

    # Create a DataFrame with the input features
    input_data = pd.DataFrame([input_data])

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