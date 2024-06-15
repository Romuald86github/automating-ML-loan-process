# run_project.py

from src.data.data_loader import generate_loan_data, load_raw_data
from src.data.data_cleaning import clean_data, save_cleaned_data
from src.features.feature_selection import load_and_select_features
from src.models.train_model import train_and_evaluate_models
import joblib
import subprocess

def run_project():
    """Run the entire loan prediction project end-to-end."""
    # Generate the synthetic loan data
    generate_loan_data()

    # Load the raw data
    loan_data = load_raw_data()

    # Clean the data
    cleaned_data = clean_data()
    save_cleaned_data(cleaned_data)

    # Perform feature selection
    X_train_selected, X_test_selected, y_train, y_test, selected_features = load_and_select_features()

    # Train the model
    best_model, preprocessing_pipeline, selected_feature_names = train_and_evaluate_models()

    # Save the models
    joblib.dump(best_model, 'models/best_performing_model.joblib')
    joblib.dump(preprocessing_pipeline, 'models/preprocessing_pipeline.joblib')
    joblib.dump(selected_feature_names, 'models/selected_feature_names.joblib')

    # Run the Streamlit app
    subprocess.run(["streamlit", "run", "src/app/app.py"])

if __name__ == "__main__":
    run_project()
