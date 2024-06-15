import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.data.data_cleaning import clean_data

def create_preprocessing_pipeline(selected_features):
    """Create a preprocessing pipeline for the loan data."""
    categorical_features = [feature for feature in selected_features if feature in ['employment_status', 'home_ownership', 'purpose']]
    numerical_features = [feature for feature in selected_features if feature not in categorical_features]

    # Encode categorical features
    categorical_encoder = ColumnTransformer(
        transformers=[
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Scale numerical features
    numeric_scaler = StandardScaler()

    # Combine the preprocessing steps into a pipeline
    preprocessing_pipeline = Pipeline([
        ('categorical_encoder', categorical_encoder),
        ('numeric_scaler', numeric_scaler)
    ])

    return preprocessing_pipeline

def preprocess_data():
    """Preprocess the loan data using the created pipeline."""
    cleaned_data = clean_data()
    selected_features = ['loan_amount', 'interest_rate', 'term', 'credit_score', 'annual_income', 'employment_status']

    preprocessing_pipeline = create_preprocessing_pipeline(selected_features)

    # Split the data into features and target
    X = cleaned_data[selected_features]
    y = cleaned_data['loan_status']

    # Apply the preprocessing pipeline
    X_preprocessed = preprocessing_pipeline.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessing_pipeline

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessing_pipeline = preprocess_data()
    print("Preprocessed Data Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
