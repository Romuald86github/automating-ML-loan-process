import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
from src.features.feature_engineering import engineer_features

def train_and_evaluate_models():
    """Train and evaluate multiple models, perform hyperparameter tuning, and return the best-performing model."""
    X_train, X_test, y_train, y_test, preprocessing_pipeline, categorical_encoder, selected_feature_names = engineer_features()

    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        MLPClassifier()
    ]

    best_model = None
    best_score = 0

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Approved')
        recall = recall_score(y_test, y_pred, pos_label='Approved')
        f1 = f1_score(y_test, y_pred, pos_label='Approved')
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print(f"Model: {type(model).__name__}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print()

        if roc_auc > best_score:
            best_model = model
            best_score = roc_auc

    # Perform hyperparameter tuning on the best model
    if isinstance(best_model, LogisticRegression):
        param_grid = {'C': [0.1, 1, 10]}
        tuned_model = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')
    elif isinstance(best_model, DecisionTreeClassifier):
        param_grid = {'max_depth': [3, 5], 'min_samples_split': [2, 5]}
        tuned_model = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')
    elif isinstance(best_model, RandomForestClassifier):
        param_grid = {'n_estimators': [100, 150], 'max_depth': [5, 10]}
        tuned_model = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')
    elif isinstance(best_model, GradientBoostingClassifier):
        param_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
        tuned_model = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')
    elif isinstance(best_model, MLPClassifier):
        param_grid = {'hidden_layer_sizes': [(10,), (20,)], 'alpha': [0.0001, 0.01]}
        tuned_model = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc')

    tuned_model.fit(X_train, y_train)
    print("Best Hyperparameters:", tuned_model.best_params_)
    print("Best ROC AUC Score:", tuned_model.best_score_)

    return tuned_model, preprocessing_pipeline, selected_feature_names

if __name__ == "__main__":
    best_model, preprocessing_pipeline, selected_feature_names = train_and_evaluate_models()
    joblib.dump(best_model, 'models/best_performing_model.joblib')
    joblib.dump(preprocessing_pipeline, 'models/preprocessing_pipeline.joblib')
    joblib.dump(selected_feature_names, 'models/selected_feature_names.joblib')
    print("Best Model, Preprocessing Pipeline, and Selected Feature Names saved.")