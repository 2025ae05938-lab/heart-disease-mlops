"""Steps 2-4: Build features, train models, track with MLflow, save the best one.

Run from the project root:
    python src/train.py
Then view the runs with:
    mlflow ui
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from data_utils import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_clean,
    split_xy,
)

RANDOM_STATE = 42  # fixed seed so results are reproducible
MODEL_PATH = os.path.join("models", "model.joblib")  # final saved pipeline
METRICS_PATH = os.path.join("models", "metrics.json")  # final metrics summary
PLOT_DIR = os.path.join("screenshots", "mlflow_plots")  # ROC curves land here


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer(
        transformers=[
            # StandardScaler: rescale to mean 0, std 1 so features are comparable
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # OneHotEncoder: turn each category into its own 0/1 column.
            # handle_unknown="ignore" stops crashes on unseen categories.
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Chain preprocessing + model so they are saved and used together."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def run_experiment(name, pipeline, X_train, X_test, y_train, y_test, params):
    """Train one model, cross-validate it, and log everything to MLflow."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]

    with mlflow.start_run(run_name=name):  # everything inside = one MLflow run
        # 5-fold cross-validation on the training data
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        cv_metrics = {f"cv_{m}": float(cv_res[f"test_{m}"].mean()) for m in scoring}

        pipeline.fit(X_train, y_train)  # final fit on all training data
        proba = pipeline.predict_proba(X_test)[:, 1]  # P(disease) on the test set
        pred = pipeline.predict(X_test)

        # Hold-out test metrics (data the model has never seen)
        test_metrics = {
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred)),
            "test_recall": float(recall_score(y_test, pred)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
        }

        # --- log to MLflow ---
        mlflow.log_params(params)  # hyper-parameters
        mlflow.log_metrics({**cv_metrics, **test_metrics})  # all numbers

        # Save and log the ROC curve plot as an artifact
        os.makedirs(PLOT_DIR, exist_ok=True)
        RocCurveDisplay.from_predictions(y_test, proba, name=name)
        plt.title(f"ROC curve - {name}")
        plot_path = os.path.join(PLOT_DIR, f"roc_{name}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(plot_path)

        # Log the whole pipeline (preprocessing + model) as an MLflow model
        # serialization_format="cloudpickle" is used because newer MLflow versions
        # default to a stricter format (skops) that rejects Random Forest trees.
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", serialization_format="cloudpickle"
        )

        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))
    return pipeline, {**cv_metrics, **test_metrics}


def main() -> None:
    df = load_clean()
    X, y = split_xy(df)
    # Hold out 20% of the data for a final test; stratify keeps class ratio equal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("heart-disease-classification")  # groups the runs

    # ---- Model 1: Logistic Regression (simple, interpretable baseline) ----
    lr_params = {"model_type": "logistic_regression", "C": 1.0, "max_iter": 1000}
    lr = build_pipeline(
        LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    )
    lr, lr_metrics = run_experiment(
        "logistic_regression", lr, X_train, X_test, y_train, y_test, lr_params
    )

    # ---- Model 2: Random Forest, tuned with a small grid search ----
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, None],
    }
    rf_base = build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    search = GridSearchCV(
        rf_base, grid, cv=5, scoring="roc_auc", n_jobs=-1  # pick best by ROC-AUC
    )
    search.fit(X_train, y_train)
    print("Best Random Forest params:", search.best_params_)
    rf_params = {"model_type": "random_forest", **search.best_params_}
    rf, rf_metrics = run_experiment(
        "random_forest", search.best_estimator_, X_train, X_test,
        y_train, y_test, rf_params,
    )

    # ---- Pick the winner by cross-validated ROC-AUC and save it ----
    if rf_metrics["cv_roc_auc"] >= lr_metrics["cv_roc_auc"]:
        best_name, best_model, best_metrics = "random_forest", rf, rf_metrics
    else:
        best_name, best_model, best_metrics = "logistic_regression", lr, lr_metrics

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)  # the file the API will load
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, **best_metrics}, f, indent=2)
    print(f"Saved best model ({best_name}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
