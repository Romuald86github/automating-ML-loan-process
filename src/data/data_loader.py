import numpy as np
import pandas as pd
from faker import Faker

def generate_loan_data():
    """Generate synthetic loan data with 12 columns and 5300 rows."""
    fake = Faker()
    columns = ['loan_id', 'loan_amount', 'interest_rate', 'term', 'credit_score', 'annual_income', 'employment_status', 'home_ownership', 'purpose', 'state', 'loan_status', 'preferred_loan_date']
    data = []
    for _ in range(5300):
        loan_id = fake.uuid4()
        loan_amount = fake.random_int(min=5000, max=100000)
        interest_rate = fake.random.uniform(0.05, 0.25)
        term = fake.random_int(min=12, max=60)
        credit_score = fake.random_int(min=500, max=850)
        annual_income = fake.random_int(min=20000, max=150000)
        employment_status = fake.random_element(elements=('Employed', 'Self-employed', 'Unemployed'))
        home_ownership = fake.random_element(elements=('Own', 'Rent', 'Mortgage'))
        purpose = fake.random_element(elements=('Debt consolidation', 'Home improvement', 'Business', 'Personal'))
        state = fake.state_abbr()
        loan_status = fake.random_element(elements=('Approved', 'Rejected'))
        preferred_loan_date = fake.date_this_decade()
        data.append([loan_id, loan_amount, interest_rate, term, credit_score, annual_income, employment_status, home_ownership, purpose, state, loan_status, preferred_loan_date])
    loan_data = pd.DataFrame(data, columns=columns)
    loan_data.to_csv('data/raw/loan_data.csv', index=False)

def load_raw_data():
    """Load the raw loan data from the data/raw directory."""
    return pd.read_csv('data/raw/loan_data.csv')
