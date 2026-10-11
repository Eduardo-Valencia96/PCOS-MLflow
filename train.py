import argparse

import mlflow

from pcos.data import load_data
from pcos.evaluate import evaluate_model, save_confusion_matrix
from pcos.preprocess import clean_data, preprocess_data, split_data
from pcos.train import train_model
from pathlib import Path


DATA_PATH = "data/raw/PCOS_data_without_infertility.xlsx"


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the PCOS classification model."
    )

    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)

    args = parser.parse_args()

    mlflow.set_experiment("PCOS Logistic Regression")

    with mlflow.start_run():
        mlflow.log_params({
            "C": args.C,
            "max_iter": args.max_iter,
            "solver": args.solver,
            "random_state": args.random_state,
            "test_size": args.test_size,
        })

        df = load_data(DATA_PATH)

        X, y = clean_data(df)

        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        X_train_processed, X_test_processed, _ = preprocess_data(
            X_train,
            X_test,
        )

        model = train_model(
            X_train_processed,
            y_train,
            C=args.C,
            max_iter=args.max_iter,
            solver=args.solver,
            random_state=args.random_state,
        )
        
        metrics, y_pred = evaluate_model(
            model,
            X_test_processed,
            y_test,
        )

        mlflow.log_metrics(metrics)

        artifact_path = Path("confusion_matrix.png")

        save_confusion_matrix(
            y_test,
            y_pred,
            artifact_path,
        )

        mlflow.log_artifact(
            artifact_path,
            artifact_path="evaluation",
        )

        artifact_path.unlink()

        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_regression_model",
        )

        print("\nModel metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()