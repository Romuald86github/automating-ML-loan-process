import pandas as pd
from src.data.data_loader import load_raw_data

def clean_data():
    """Clean the loan data."""
    loan_data = load_raw_data()
    
    # Handle inconsistent data types
    loan_data['loan_amount'] = loan_data['loan_amount'].astype('int64')
    loan_data['interest_rate'] = loan_data['interest_rate'].astype('float64')
    loan_data['term'] = loan_data['term'].astype('int64')
    loan_data['credit_score'] = loan_data['credit_score'].astype('int64')
    loan_data['annual_income'] = loan_data['annual_income'].astype('int64')

    # Handle missing values
    loan_data = loan_data.dropna()

    # Remove duplicates
    loan_data = loan_data.drop_duplicates()

    # Ensure consistent column names
    loan_data.columns = [col.lower().replace(' ', '_') for col in loan_data.columns]

    return loan_data

def save_cleaned_data(cleaned_data):
    """Save the cleaned loan data to the data/processed directory."""
    cleaned_data.to_csv('data/processed/cleaned_loan_data.csv', index=False)

if __name__ == "__main__":
    cleaned_data = clean_data()
    save_cleaned_data(cleaned_data)
