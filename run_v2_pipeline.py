"""
V2 Recommendation Pipeline Execution
====================================
Upgrades the Ranking Layer from V1 (Logistic Regression) to V2 (XGBoost GBDT Ranker).
Evaluates and benchmarks:
    - Baseline: Popularity Model
    - V1: Logistic Regression Ranker
    - V2: XGBoost GBDT Ranker

Logs all metrics, model parameters, and artifacts to MLflow under 'recsys_v2'.
"""

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

    print("=" * 75)
    print("RETAILROCKET RECOMMENDATION SYSTEM — V2 PIPELINE (XGBOOST GBDT RANKER)")
    print("=" * 75)

    # 1. Load Data
    print("\n[1/6] Loading Train and Test Data...")
    train = pd.read_parquet(processed_dir / "train_events.parquet")
    test = pd.read_parquet(processed_dir / "test_events.parquet")
    print(f"  Train: {len(train):,} events | Test: {len(test):,} events")

    # 2. Features (load from parquet cache if present)
    print("\n[2/6] Loading Feature Store...")
    user_feat_path = features_dir / "user_features.parquet"
    item_feat_path = features_dir / "item_features.parquet"
    inter_feat_path = features_dir / "interaction_features.parquet"

    if user_feat_path.exists() and item_feat_path.exists() and inter_feat_path.exists():
        print("  Loading precomputed feature stores from parquet...")
        user_features = pd.read_parquet(user_feat_path)
        item_features = pd.read_parquet(item_feat_path)
        interaction_features = pd.read_parquet(inter_feat_path)
    else:
        print("  Building feature stores...")
        user_features = build_user_features(train)
        item_features = build_item_features(train, config)
        interaction_features = build_interaction_features(train)
        features_dir.mkdir(parents=True, exist_ok=True)
        user_features.to_parquet(user_feat_path, index=False)
        item_features.to_parquet(item_feat_path, index=False)
        interaction_features.to_parquet(inter_feat_path, index=False)

    print(f"  User features: {len(user_features):,} users")
    print(f"  Item features: {len(item_features):,} items")
    print(f"  Interaction features: {len(interaction_features):,} pairs")

    # 3. Popularity Baseline & Item-CF
    print("\n[3/6] Generating Candidates & Models...")
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

    # 4. Build Ranking Dataset & Train Models
    print("\n[4/6] Building Ranking Dataset...")
    ranking_data = build_ranking_dataset(
        candidates, test, user_features, item_features, interaction_features, config
    )

    print("\n--- Training V1: Logistic Regression ---")
    lr_config = dict(config)
    lr_config["mlflow"]["experiment_name"] = "recsys_v2"
    lr_pipeline, lr_feature_cols = train_ranking_model(
        ranking_data, lr_config, model_type="logistic_regression"
    )

    print("\n--- Training V2: XGBoost GBDT ---")
    xgb_config = dict(config)
    xgb_config["mlflow"]["experiment_name"] = "recsys_v2"
    xgb_pipeline, xgb_feature_cols = train_ranking_model(
        ranking_data, xgb_config, model_type="xgboost"
    )

    # 5. Fast Batch Evaluation
    print("\n[5/6] High-Speed Batch Evaluation (5,000 Users)...")
    test_user_items = (
        test[test["visitorid"].isin(eval_users_sample)]
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )
    total_items = train["itemid"].nunique()

    # Model 0: Popularity
    print("  Scoring Baseline: Popularity...")
    pop_recs = {u: popularity_recs for u in eval_users_sample}
    pop_metrics = evaluate_recommendations(pop_recs, test_user_items, k, total_items)
    pop_biz = compute_business_metrics(pop_recs, test, k)

    # Model 1: V1 Logistic Regression (Batch)
    print("  Scoring V1: Logistic Regression...")
    t0 = time.time()
    lr_recs = predict_for_users_batch(
        eval_users_sample, candidates, user_features, item_features,
        interaction_features, lr_pipeline, lr_feature_cols, k=k
    )
    print(f"    V1 batch inference completed in {time.time() - t0:.2f}s")
    lr_metrics = evaluate_recommendations(lr_recs, test_user_items, k, total_items)
    lr_biz = compute_business_metrics(lr_recs, test, k)

    # Model 2: V2 XGBoost (Batch)
    print("  Scoring V2: XGBoost GBDT...")
    t0 = time.time()
    xgb_recs = predict_for_users_batch(
        eval_users_sample, candidates, user_features, item_features,
        interaction_features, xgb_pipeline, xgb_feature_cols, k=k
    )
    print(f"    V2 batch inference completed in {time.time() - t0:.2f}s")
    xgb_metrics = evaluate_recommendations(xgb_recs, test_user_items, k, total_items)
    xgb_biz = compute_business_metrics(xgb_recs, test, k)

    # 6. Log Comparison to MLflow
    print("\n[6/6] Logging Benchmark Results to MLflow...")
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment("recsys_v2")

        with mlflow.start_run(run_name="v2_model_comparison_benchmark"):
            for model_prefix, metrics in [
                ("pop", pop_metrics),
                ("v1_lr", lr_metrics),
                ("v2_xgb", xgb_metrics),
            ]:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        clean_key = f"{model_prefix}_{metric_name}".replace("@", "_at_")
                        mlflow.log_metric(clean_key, value)

            for model_prefix, biz in [
                ("pop", pop_biz),
                ("v1_lr", lr_biz),
                ("v2_xgb", xgb_biz),
            ]:
                for metric_name, value in biz.items():
                    if isinstance(value, (int, float)):
                        clean_key = f"{model_prefix}_{metric_name}".replace("@", "_at_")
                        mlflow.log_metric(clean_key, value)

        print("  Comparison benchmark successfully logged to MLflow under 'recsys_v2'!")
    except Exception as e:
        print(f"  Warning: MLflow logging skipped ({e})")

    # Print Comparison Table
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SCORECARD: POPULARITY vs. V1 (LR) vs. V2 (XGBOOST GBDT)")
    print("=" * 80)
    header = f"{'Metric':<18} {'Popularity':<14} {'V1 (Logistic)':<16} {'V2 (XGBoost)':<16} {'V2 vs V1 Lift':<14}"
    print(header)
    print("-" * 80)

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

        if val_lr > 0:
            lift = ((val_xgb - val_lr) / val_lr) * 100
            lift_str = f"{lift:+.1f}%"
        else:
            lift_str = "N/A"

        print(f"{m:<18} {val_pop:<14.6f} {val_lr:<16.6f} {val_xgb:<16.6f} {lift_str:<14}")

    print("-" * 80)
    print("BUSINESS CONVERSION METRICS:")
    for b in ["view_rate", "cart_rate", "purchase_rate"]:
        val_pop = pop_biz[b]
        val_lr = lr_biz[b]
        val_xgb = xgb_biz[b]
        if val_lr > 0:
            lift = ((val_xgb - val_lr) / val_lr) * 100
            lift_str = f"{lift:+.1f}%"
        else:
            lift_str = "N/A"
        print(f"{b:<18} {val_pop:<14.6f} {val_lr:<16.6f} {val_xgb:<16.6f} {lift_str:<14}")

    print("=" * 80)
    print(f"Total V2 pipeline execution time: {elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
