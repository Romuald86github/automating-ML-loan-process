import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from src.features.preprocessing import preprocess_data

def engineer_features():
    """Perform feature engineering on the loan data."""
    X_train, X_test, y_train, y_test, preprocessing_pipeline = preprocess_data()

    # Original feature names before transformation
    original_feature_names = [
        'loan_amount', 'interest_rate', 'term', 'credit_score', 'annual_income', 
        'employment_status', 'home_ownership', 'purpose'
    ]

    # Feature selection using SelectKBest
    selector = SelectKBest(f_classif, k=6)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get the indices of the selected features
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [original_feature_names[i] for i in selected_feature_indices]

    print("Selected Features:", selected_feature_names)

    return X_train_selected, X_test_selected, y_train, y_test, preprocessing_pipeline, selected_feature_names

if __name__ == "__main__":
    X_train_selected, X_test_selected, y_train, y_test, preprocessing_pipeline, selected_feature_names = engineer_features()
    print("Engineered Data Shapes:", X_train_selected.shape, X_test_selected.shape, y_train.shape, y_test.shape)
    print("Selected Features:", selected_feature_names)