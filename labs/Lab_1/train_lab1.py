import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def add_combined_feature(X):
    X = X.copy()  # Ensure we're modifying a copy of the DataFrame
    # Example feature: combining two features
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    return X

def main():
    print("1. Loading Breast Cancer dataset...")
    raw_data = load_breast_cancer(as_frame=True)
    df = pd.concat([raw_data['data'], raw_data['target']], axis=1)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Data split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")

    # Define the feature engineering and preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('feature_engineering', FunctionTransformer(add_combined_feature)),
        ('scaler', StandardScaler())
    ])

    # Base pipeline structure
    training_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('classifier', LogisticRegression()) # Placeholder
    ])

    # Grid search parameters for multiple models as requested in Lab 1 exercise:
    # 1. Logistic Regression
    # 2. Random Forest (n_estimators: 50, 100, 200; max_depth: None, 10, 20)
    # 3. SVC (C: 0.1, 1, 10; kernel: linear, rbf)
    param_grid = [
        {
            'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0]
        },
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        },
        {
            'classifier': [SVC(random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['linear', 'rbf']
        }
    ]

    print("\n2. Running GridSearchCV across Logistic Regression, Random Forest, and SVC...")
    grid_search = GridSearchCV(training_pipeline, param_grid, cv=5, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("\n--- Search Results ---")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    print("\n3. Evaluating best model on test set...")
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    output_path = 'best_cancer_model_pipeline.joblib'
    joblib.dump(best_model, output_path)
    print(f"\n4. Saved best model pipeline to '{output_path}'.")

if __name__ == '__main__':
    main()
