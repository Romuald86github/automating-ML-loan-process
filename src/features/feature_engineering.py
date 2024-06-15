import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from src.features.preprocessing import preprocess_data

def engineer_features():
    """Perform feature engineering on the loan data."""
    X_train, X_test, y_train, y_test, preprocessing_pipeline, categorical_encoder = preprocess_data()

    # Feature selection using SelectKBest
    selector = SelectKBest(f_classif, k=6)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get the selected feature names
    selected_feature_names = categorical_encoder.get_feature_names_out()
    selected_feature_names = list(selected_feature_names) + ['loan_amount', 'interest_rate', 'term', 'credit_score', 'annual_income']

    return X_train_selected, X_test_selected, y_train, y_test, preprocessing_pipeline, categorical_encoder, selected_feature_names

if __name__ == "__main__":
    X_train_selected, X_test_selected, y_train, y_test, preprocessing_pipeline, categorical_encoder, selected_feature_names = engineer_features()
    print("Engineered Data Shapes:", X_train_selected.shape, X_test_selected.shape, y_train.shape, y_test.shape)
    print("Selected Features:", selected_feature_names)