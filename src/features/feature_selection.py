# src/features/feature_selection.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_select_features():
    """Load cleaned data and select specified features."""
    cleaned_data = pd.read_csv('data/processed/cleaned_loan_data.csv')
    
    selected_features = ['loan_amount', 'term', 'credit_score', 'annual_income', 'home_ownership', 'purpose', 'employment_status', 'interest_rate']
    X = cleaned_data[selected_features]
    y = cleaned_data['loan_status']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, selected_features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, selected_features = load_and_select_features()
    print("Selected Features:", selected_features)
    print("Train and Test Data Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
