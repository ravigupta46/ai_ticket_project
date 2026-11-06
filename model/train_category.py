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

    models = [
        (
            "Multinomial Naive Bayes",
            MultinomialNB(),
            {
                'classifier__alpha': [0.01, 0.1, 0.5, 1.0]
            }
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=42),
            {
                'classifier__C': [0.5, 1.0, 3.0, 5.0, 10.0]
            }
        ),
        (
            "Linear SVC",
            LinearSVC(max_iter=3000, random_state=42),
            {
                'classifier__C': [0.5, 0.8, 1.0, 1.2]
            }
        )
    ]

    best_model = None
    best_score = -1
    best_params = None
    best_name = None

    for name, clf, params in models:

        pipe = Pipeline([
            ('features', create_text_feature_union()),
            ('classifier', clf)
        ])

        grid = GridSearchCV(
            pipe,
            param_grid=params,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        print(f"{name}: {grid.best_score_:.4f}")
        print(f"Best Params: {grid.best_params_}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            best_name = name

    print(f"\nBest Category Model: {best_name}")
    print(f"Best Weighted F1: {best_score:.4f}")

    return best_model, best_params, best_score



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
