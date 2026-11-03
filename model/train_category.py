"""model/train_category.py
Train, tune, and persist the CATEGORY classification pipeline.
"""
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.config import CAT_MODEL_PATH
from model.preprocessing import clean_data, preprocess_and_split, create_text_feature_union


def build_and_compare_category_models(X_train, X_test, y_train, y_test):
    """Train MultinomialNB, LogisticRegression, LinearSVC on the 
    FeatureUnion. Print and return a DataFrame of metrics."""
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=2000, random_state=42),
        "Linear Support Vector (SVC)": LinearSVC(C=0.8, max_iter=3000, random_state=42)
    }

    records = []
    for name, clf in models.items():
        pipe = Pipeline([
            ('features', create_text_feature_union()),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        records.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, preds, average='weighted', zero_division=0)
        })

    metrics_df = pd.DataFrame(records)
    print("\n--- Category Models Comparison ---")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return metrics_df


def tune_category_model(X_train, y_train):
    """GridSearchCV on LinearSVC with C in [0.5, 0.8, 1.0, 1.2].
    Return best_estimator, best_params, best_score."""
    pipe = Pipeline([
        ('features', create_text_feature_union()),
        ('classifier', LinearSVC(max_iter=3000, random_state=42))
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={'classifier__C': [0.5, 0.8, 1.0, 1.2]},
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best Category Params: {grid.best_params_} (CV F1: {grid.best_score_:.4f})")
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_category_pipeline(df=None, splits=None):
    """Full flow: clean_data -> preprocess_and_split -> 
    build_and_compare_category_models -> tune_category_model -> 
    pickle.dump to CAT_MODEL_PATH -> return (best_pipe, best_params).
    Called by run_training.py."""
    if splits is not None:
        X_train, X_test, y_train, y_test = splits
    else:
        if df is None:
            df = clean_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess_and_split(df)

    build_and_compare_category_models(X_train, X_test, y_train, y_test)
    best_pipe, best_params, _ = tune_category_model(X_train, y_train)

    with open(CAT_MODEL_PATH, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"Saved Category Pipeline -> {CAT_MODEL_PATH}")

    return best_pipe, best_params


if __name__ == "__main__":
    df = clean_data()
    train_category_pipeline(df)
