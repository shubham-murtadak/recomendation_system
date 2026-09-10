"""
Ranking Model Training
======================
Trains ranking models (XGBoost, LightGBM, or Logistic Regression)
on the candidate dataset.
Logs all hyperparameters, feature importances, and metrics to MLflow.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

from src.ranking.dataset import get_feature_columns


def train_ranking_model(
    dataset: pd.DataFrame,
    config: dict,
    model_type: str = "xgboost",
) -> tuple:
    """
    Train ranking model: XGBoost (V2 GBDT), LightGBM (V2 GBDT), or Logistic Regression (V1).
    
    Returns:
        (pipeline, feature_columns)
    """
    feature_cols = get_feature_columns(dataset)
    X = np.ascontiguousarray(dataset[feature_cols].values, dtype=np.float32)
    y = np.ascontiguousarray(dataset["label"].values, dtype=np.int32)

    model_type_clean = model_type.lower()
    print(f"Training ranking model ({model_type_clean.upper()})...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,}")
    print(f"  Positive rate: {y.mean():.4f}")

    random_state = config["ranking"].get("random_state", 42)

    if model_type_clean in ("xgboost", "xgb"):
        # XGBoost GBDT Ranker
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos = float(neg_count) / max(pos_count, 1)

        xgb_params = {
            "n_estimators": config["ranking"].get("n_estimators", 150),
            "learning_rate": config["ranking"].get("learning_rate", 0.05),
            "max_depth": config["ranking"].get("max_depth", 5),
            "subsample": config["ranking"].get("subsample", 0.8),
            "colsample_bytree": config["ranking"].get("colsample_bytree", 0.8),
            "scale_pos_weight": scale_pos,
            "random_state": random_state,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        classifier = xgb.XGBClassifier(**xgb_params)
        pipeline = Pipeline([("classifier", classifier)])
        pipeline.fit(X, y)

        train_score = pipeline.score(X, y)
        print(f"  Training accuracy: {train_score:.4f}")

        importances = classifier.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        print("\n  Top 10 features by importance (XGBoost GBDT):")
        print(importance_df.head(10).to_string(index=False))

        run_name = "v2_xgboost_ranker"
        model_params_to_log = xgb_params

    elif model_type_clean in ("lightgbm", "lgbm"):
        # LightGBM GBDT Ranker
        lgb_params = {
            "n_estimators": config["ranking"].get("n_estimators", 150),
            "learning_rate": config["ranking"].get("learning_rate", 0.05),
            "num_leaves": config["ranking"].get("num_leaves", 31),
            "max_depth": config["ranking"].get("max_depth", 6),
            "subsample": config["ranking"].get("subsample", 0.8),
            "colsample_bytree": config["ranking"].get("colsample_bytree", 0.8),
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        classifier = lgb.LGBMClassifier(**lgb_params)
        pipeline = Pipeline([("classifier", classifier)])
        pipeline.fit(X, y)

        train_score = pipeline.score(X, y)
        print(f"  Training accuracy: {train_score:.4f}")

        importances = classifier.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        print("\n  Top 10 features by importance (LightGBM):")
        print(importance_df.head(10).to_string(index=False))

        run_name = "v2_lightgbm_ranker"
        model_params_to_log = lgb_params

    elif model_type_clean in ("deepfm", "deep_fm"):
        # DeepFM Deep Ranking Model (V3)
        from src.ranking.deepfm import DeepFMRanker
        deepfm_params = {
            "embedding_dim": config["ranking"].get("embedding_dim", 16),
            "hidden_dims": config["ranking"].get("hidden_dims", [128, 64, 32]),
            "dropout": config["ranking"].get("dropout", 0.2),
            "lr": config["ranking"].get("lr", 0.002),
            "epochs": config["ranking"].get("epochs", 25),
            "batch_size": config["ranking"].get("batch_size", 128),
            "random_state": random_state,
        }
        pipeline = DeepFMRanker(**deepfm_params)
        pipeline.fit(X, y)

        train_score = pipeline.score(X, y)
        print(f"  Training accuracy: {train_score:.4f}")

        run_name = "v3_deepfm_ranker"
        model_params_to_log = deepfm_params

    else:
        # Logistic Regression Baseline (V1)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])
        pipeline.fit(X, y)

        train_score = pipeline.score(X, y)
        print(f"  Training accuracy: {train_score:.4f}")

        coefficients = pipeline.named_steps["classifier"].coef_[0]
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }).sort_values("abs_coefficient", ascending=False)

        print("\n  Top 10 features by importance (Logistic Regression):")
        print(importance_df.head(10).to_string(index=False))

        run_name = "v1_logistic_regression"
        model_params_to_log = {"solver": "lbfgs", "class_weight": "balanced"}

    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_filename = f"ranking_model_{model_type_clean}.joblib"
    model_path = models_dir / model_filename
    joblib.dump(pipeline, model_path)

    # Save as default ranking_model.joblib
    joblib.dump(pipeline, models_dir / "ranking_model.joblib")

    feature_path = models_dir / "feature_columns.joblib"
    joblib.dump(feature_cols, feature_path)

    print(f"\n  Model saved to {model_path} (and ranking_model.joblib)")
    print(f"  Feature list saved to {feature_path}")

    # Log to MLflow
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"].get("experiment_name", "recsys_v2"))

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", model_type_clean)
            mlflow.log_param("features_count", len(feature_cols))
            mlflow.log_param("train_samples", len(X))
            mlflow.log_param("positive_rate", round(float(y.mean()), 4))
            mlflow.log_param("negative_sample_ratio", config["ranking"].get("negative_sample_ratio", 5))

            for param_k, param_v in model_params_to_log.items():
                mlflow.log_param(param_k, param_v)

            mlflow.log_metric("train_accuracy", float(train_score))
            mlflow.log_artifact(str(model_path))
            print("  MLflow run logged successfully.")
    except Exception as e:
        print(f"  Warning: MLflow logging skipped ({e})")

    return pipeline, feature_cols
