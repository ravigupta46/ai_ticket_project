"""model/run_training.py
Main entry point to execute the full ML training and evaluation workflow.
"""
from model.config import DATA_PATH
from model.preprocessing import clean_data, preprocess_and_split
from model.eda import perform_eda
from model.train_category import train_category_pipeline
from model.train_priority import train_priority_pipeline
from model.evaluate import evaluate_and_save_metrics


def main():
    print("STEP 1: Cleaning...")
    df = clean_data(DATA_PATH)

    print("STEP 2: EDA...")
    perform_eda(df)

    print("STEP 3: Preprocessing...")
    X_train, X_test, y_cat_tr, y_cat_te, y_pri_tr, y_pri_te = preprocess_and_split(df)

    print("STEP 4: Train category...")
    cat_pipe, cat_params = train_category_pipeline(splits=(X_train, X_test, y_cat_tr, y_cat_te))

    print("STEP 5: Train priority...")
    pri_pipe, pri_params = train_priority_pipeline(splits=(X_train, X_test, y_pri_tr, y_pri_te))

    print("STEP 6: Evaluate...")
    evaluate_and_save_metrics(
        cat_pipe, pri_pipe, X_test,
        y_cat_te, y_pri_te,
        cat_params, pri_params
    )
    print("\nDONE. Models saved. Metrics written.")


if __name__ == "__main__":
    main()
