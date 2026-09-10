"""
V3 Recommendation Pipeline Execution: Deep Recommendation (DeepFM)
==================================================================
Implements and evaluates V3 of the Recommendation Architecture.
Benchmarks all four generations of models side-by-side on 5,000 test users:
    - Baseline: Popularity Model
    - V1: Logistic Regression Ranker
    - V2: XGBoost GBDT Ranker
    - V3: DeepFM Deep Neural Network Ranker

Logs all parameters, metrics, and models to MLflow under 'recsys_v3'.
"""

import torch
import sys
import os
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import mlflow

# Fix Windows console utf-8 encoding for MLflow output
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.features.interaction_features import build_interaction_features
from src.candidate_generation.popularity import (
    build_popularity_ranking,
    get_popular_candidates,
    recommend_popular,
)
from src.candidate_generation.item_cf import run_item_cf
from src.candidate_generation.candidate_service import generate_candidates_batch
from src.ranking.dataset import build_ranking_dataset
from src.ranking.train import train_ranking_model
from src.ranking.predict import predict_for_users_batch
from src.evaluation.ranking_metrics import evaluate_recommendations
from src.evaluation.business_metrics import compute_business_metrics


def load_config() -> dict:
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    k = config["evaluation"]["top_k"]

    start_time = time.time()

    print("=" * 82)
    print("RETAILROCKET RECOMMENDATION SYSTEM — V3 PIPELINE (DEEPFM DEEP RECOMMENDATION)")
    print("=" * 82)

    # 1. Load Data
    print("\n[1/6] Loading Train and Test Data...")
    train = pd.read_parquet(processed_dir / "train_events.parquet")
    test = pd.read_parquet(processed_dir / "test_events.parquet")
    print(f"  Train: {len(train):,} events | Test: {len(test):,} events")

    # 2. Features (load from parquet cache)
    print("\n[2/6] Loading Feature Store...")
    user_features = pd.read_parquet(features_dir / "user_features.parquet")
    item_features = pd.read_parquet(features_dir / "item_features.parquet")
    interaction_features = pd.read_parquet(features_dir / "interaction_features.parquet")

    print(f"  User features: {len(user_features):,} users")
    print(f"  Item features: {len(item_features):,} items")
    print(f"  Interaction features: {len(interaction_features):,} pairs")

    # 3. Popularity Baseline & Item-CF
    print("\n[3/6] Generating Candidates & Seed Models...")
    popularity = build_popularity_ranking(train, config)
    popular_items = get_popular_candidates(
        popularity, config["candidate_generation"]["popularity_top_n"]
    )
    popularity_recs = recommend_popular(popularity, k=k)

    similarity_dict = run_item_cf(config, force_recompute=False)

    # Eval users
    train_users = set(train["visitorid"].unique())
    test_users_all = set(test["visitorid"].unique())
    eval_users = list(train_users & test_users_all)
    max_eval_users = min(len(eval_users), 5000)
    np.random.seed(config["ranking"]["random_state"])
    eval_users_sample = list(np.random.choice(eval_users, max_eval_users, replace=False))
    print(f"  Evaluation cohort: {len(eval_users_sample):,} users")

    candidates = generate_candidates_batch(
        eval_users_sample, train, similarity_dict, popular_items, config
    )

    # 4. Build Ranking Dataset & Train All Three Model Generations
    print("\n[4/6] Building Ranking Dataset...")
    ranking_data = build_ranking_dataset(
        candidates, test, user_features, item_features, interaction_features, config
    )

    v3_config = dict(config)
    v3_config["mlflow"]["experiment_name"] = "recsys_v3"

    print("\n--- Training V1: Logistic Regression ---")
    lr_pipeline, lr_feature_cols = train_ranking_model(
        ranking_data, v3_config, model_type="logistic_regression"
    )

    print("\n--- Training V2: XGBoost GBDT ---")
    xgb_pipeline, xgb_feature_cols = train_ranking_model(
        ranking_data, v3_config, model_type="xgboost"
    )

    print("\n--- Training V3: DeepFM (Deep Neural Network + FM) ---")
    deepfm_pipeline, deepfm_feature_cols = train_ranking_model(
        ranking_data, v3_config, model_type="deepfm"
    )

    # 5. Fast Batch Evaluation Across All 4 Models
    print("\n[5/6] High-Speed Batch Evaluation (5,000 Users across 4 Models)...")
    test_user_items = (
        test[test["visitorid"].isin(eval_users_sample)]
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )
    total_items = train["itemid"].nunique()

    # Model 0: Popularity
    print("  [Model 0] Scoring Baseline: Popularity...")
    pop_recs = {u: popularity_recs for u in eval_users_sample}
    pop_metrics = evaluate_recommendations(pop_recs, test_user_items, k, total_items)
    pop_biz = compute_business_metrics(pop_recs, test, k)

    # Model 1: V1 Logistic Regression (Batch)
    print("  [Model 1] Scoring V1: Logistic Regression...")
    t0 = time.time()
    lr_recs = predict_for_users_batch(
        eval_users_sample, candidates, user_features, item_features,
        interaction_features, lr_pipeline, lr_feature_cols, k=k
    )
    print(f"    V1 batch inference completed in {time.time() - t0:.2f}s")
    lr_metrics = evaluate_recommendations(lr_recs, test_user_items, k, total_items)
    lr_biz = compute_business_metrics(lr_recs, test, k)

    # Model 2: V2 XGBoost GBDT (Batch)
    print("  [Model 2] Scoring V2: XGBoost GBDT...")
    t0 = time.time()
    xgb_recs = predict_for_users_batch(
        eval_users_sample, candidates, user_features, item_features,
        interaction_features, xgb_pipeline, xgb_feature_cols, k=k
    )
    print(f"    V2 batch inference completed in {time.time() - t0:.2f}s")
    xgb_metrics = evaluate_recommendations(xgb_recs, test_user_items, k, total_items)
    xgb_biz = compute_business_metrics(xgb_recs, test, k)

    # Model 3: V3 DeepFM Neural Network (Batch)
    print("  [Model 3] Scoring V3: DeepFM Neural Network...")
    t0 = time.time()
    deepfm_recs = predict_for_users_batch(
        eval_users_sample, candidates, user_features, item_features,
        interaction_features, deepfm_pipeline, deepfm_feature_cols, k=k
    )
    print(f"    V3 batch inference completed in {time.time() - t0:.2f}s")
    deepfm_metrics = evaluate_recommendations(deepfm_recs, test_user_items, k, total_items)
    deepfm_biz = compute_business_metrics(deepfm_recs, test, k)

    # 6. Log Comprehensive Benchmark to MLflow
    print("\n[6/6] Logging V3 Benchmark Results to MLflow...")
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment("recsys_v3")

        with mlflow.start_run(run_name="v3_grand_model_comparison"):
            for model_prefix, metrics in [
                ("pop", pop_metrics),
                ("v1_lr", lr_metrics),
                ("v2_xgb", xgb_metrics),
                ("v3_deepfm", deepfm_metrics),
            ]:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        clean_key = f"{model_prefix}_{metric_name}".replace("@", "_at_")
                        mlflow.log_metric(clean_key, value)

            for model_prefix, biz in [
                ("pop", pop_biz),
                ("v1_lr", lr_biz),
                ("v2_xgb", xgb_biz),
                ("v3_deepfm", deepfm_biz),
            ]:
                for metric_name, value in biz.items():
                    if isinstance(value, (int, float)):
                        clean_key = f"{model_prefix}_{metric_name}".replace("@", "_at_")
                        mlflow.log_metric(clean_key, value)

        print("  Comparison benchmark successfully logged to MLflow under 'recsys_v3'!")
    except Exception as e:
        print(f"  Warning: MLflow logging skipped ({e})")

    # Print 4-Way Comparison Table
    elapsed = time.time() - start_time
    print("\n" + "=" * 92)
    print("GRAND BENCHMARK: POPULARITY vs. V1 (LR) vs. V2 (XGBOOST) vs. V3 (DEEPFM)")
    print("=" * 92)
    header = f"{'Metric':<16} {'Popularity':<14} {'V1 (Logistic)':<15} {'V2 (XGBoost)':<15} {'V3 (DeepFM)':<15} {'Best Model':<14}"
    print(header)
    print("-" * 92)

    metrics_to_show = [
        f"ndcg@{k}",
        f"recall@{k}",
        f"precision@{k}",
        f"hit_rate@{k}",
        f"mrr@{k}",
        "coverage",
    ]

    for m in metrics_to_show:
        val_pop = pop_metrics[m]
        val_lr = lr_metrics[m]
        val_xgb = xgb_metrics[m]
        val_dfm = deepfm_metrics[m]

        scores = {"Popularity": val_pop, "V1 (LR)": val_lr, "V2 (XGB)": val_xgb, "V3 (DeepFM)": val_dfm}
        best = max(scores.items(), key=lambda x: x[1])[0]

        print(f"{m:<16} {val_pop:<14.6f} {val_lr:<15.6f} {val_xgb:<15.6f} {val_dfm:<15.6f} {best:<14}")

    print("-" * 92)
    print("BUSINESS CONVERSION METRICS:")
    for b in ["view_rate", "cart_rate", "purchase_rate"]:
        val_pop = pop_biz[b]
        val_lr = lr_biz[b]
        val_xgb = xgb_biz[b]
        val_dfm = deepfm_biz[b]
        scores = {"Popularity": val_pop, "V1 (LR)": val_lr, "V2 (XGB)": val_xgb, "V3 (DeepFM)": val_dfm}
        best = max(scores.items(), key=lambda x: x[1])[0]
        print(f"{b:<16} {val_pop:<14.6f} {val_lr:<15.6f} {val_xgb:<15.6f} {val_dfm:<15.6f} {best:<14}")

    print("=" * 92)
    print(f"Total V3 pipeline execution time: {elapsed:.1f} seconds")
    print("=" * 92)


if __name__ == "__main__":
    main()
