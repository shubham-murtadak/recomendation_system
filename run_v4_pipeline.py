"""
V4 Recommendation Pipeline Execution: Two-Stage Recommender with Re-Ranking Layer
================================================================================
Implements V4 Architecture:
    User
     │
     ▼
[Stage 1: Candidate Retrieval]  (~150 items: Item-CF + Popularity)
     │
     ▼
[Stage 2: Precision Ranking]    (Top 30 items: XGBoost GBDT Ranker)
     │
     ▼
[Stage 3: Re-Ranking Layer]     (Top 10 items: MMR Diversity + Category Capping)

Evaluates the accuracy vs. diversity trade-off and logs to MLflow under 'recsys_v4'.
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
from src.ranking.predict import (
    predict_for_users_batch,
    predict_ranked_candidates_batch,
)
from src.reranking.diversity import rerank_batch
from src.evaluation.ranking_metrics import evaluate_recommendations
from src.evaluation.business_metrics import compute_business_metrics
from src.evaluation.diversity_metrics import evaluate_diversity_metrics


def load_config() -> dict:
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    k = config["evaluation"]["top_k"]

    start_time = time.time()

    print("=" * 86)
    print("RETAILROCKET RECOMMENDATION SYSTEM — V4 PIPELINE (TWO-STAGE + RE-RANKING LAYER)")
    print("=" * 86)

    # 1. Load Data & Features
    print("\n[1/6] Loading Train/Test Data and Feature Stores...")
    train = pd.read_parquet(processed_dir / "train_events.parquet")
    test = pd.read_parquet(processed_dir / "test_events.parquet")

    user_features = pd.read_parquet(features_dir / "user_features.parquet")
    item_features = pd.read_parquet(features_dir / "item_features.parquet")
    interaction_features = pd.read_parquet(features_dir / "interaction_features.parquet")

    # Build category lookup map for re-ranking
    category_map = dict(zip(item_features["itemid"], item_features["item_category"]))
    total_items = train["itemid"].nunique()

    print(f"  Train events: {len(train):,} | Test events: {len(test):,}")
    print(f"  Catalog items: {total_items:,} items | Category mapping: {len(category_map):,} items")

    # 2. Stage 1: Candidate Retrieval (150 items)
    print("\n[2/6] STAGE 1: Candidate Retrieval (Item-CF + Popularity Pool)...")
    popularity = build_popularity_ranking(train, config)
    popular_items = get_popular_candidates(
        popularity, config["candidate_generation"]["popularity_top_n"]
    )
    similarity_dict = run_item_cf(config, force_recompute=False)

    train_users = set(train["visitorid"].unique())
    test_users_all = set(test["visitorid"].unique())
    eval_users = list(train_users & test_users_all)
    max_eval_users = min(len(eval_users), 5000)
    np.random.seed(config["ranking"]["random_state"])
    eval_users_sample = list(np.random.choice(eval_users, max_eval_users, replace=False))

    print(f"  Retrieving ~150 candidates for {len(eval_users_sample):,} evaluation users...")
    candidates = generate_candidates_batch(
        eval_users_sample, train, similarity_dict, popular_items, config
    )

    # 3. Stage 2: Train Ranking Model (XGBoost GBDT)
    print("\n[3/6] STAGE 2: Ranking Model Training (XGBoost GBDT)...")
    ranking_data = build_ranking_dataset(
        candidates, test, user_features, item_features, interaction_features, config
    )

    v4_config = dict(config)
    v4_config["mlflow"]["experiment_name"] = "recsys_v4"
    xgb_pipeline, xgb_feature_cols = train_ranking_model(
        ranking_data, v4_config, model_type="xgboost"
    )

    # 4. Generate Predictions: Pure Stage 2 vs. Stage 3 Re-Ranked
    print("\n[4/6] STAGE 3: Re-Ranking Layer (MMR Diversity + Category Capping)...")
    t0 = time.time()

    # Step A: Score candidates and extract top 30 candidates per user for Stage 3
    print("  Scoring candidates & selecting Top 30 for Re-Ranking...")
    ranked_candidates_top30 = predict_ranked_candidates_batch(
        eval_users_sample, candidates, user_features, item_features,
        interaction_features, xgb_pipeline, xgb_feature_cols, top_n=30
    )

    # Pure Stage 2 baseline (take top-10 directly from model scores without diversity)
    pure_stage2_recs = {
        uid: [cand["itemid"] for cand in cand_list[:k]]
        for uid, cand_list in ranked_candidates_top30.items()
    }

    # Step B: Apply Stage 3 Re-Ranking (MMR lambda=0.7, max 2 items per category)
    print("  Applying Maximal Marginal Relevance (lambda=0.7) & Category Capping (max 2/cat)...")
    v4_reranked_recs = rerank_batch(
        ranked_candidates_top30,
        k=k,
        lambda_param=0.7,
        max_per_category=2,
        similarity_map=similarity_dict,
        category_map=category_map,
    )
    print(f"  Stage 3 Re-Ranking completed in {time.time() - t0:.2f}s")

    # 5. Dual Evaluation: Accuracy + Diversity
    print("\n[5/6] Comprehensive Evaluation: Accuracy vs. Diversity Trade-Off...")
    test_user_items = (
        test[test["visitorid"].isin(eval_users_sample)]
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    # Popularity Baseline
    pop_recs = {u: recommend_popular(popularity, k=k) for u in eval_users_sample}
    pop_acc = evaluate_recommendations(pop_recs, test_user_items, k, total_items)
    pop_div = evaluate_diversity_metrics(pop_recs, category_map, similarity_dict, total_items)
    pop_biz = compute_business_metrics(pop_recs, test, k)

    # Pure Stage 2 (XGBoost Top 10)
    stage2_acc = evaluate_recommendations(pure_stage2_recs, test_user_items, k, total_items)
    stage2_div = evaluate_diversity_metrics(pure_stage2_recs, category_map, similarity_dict, total_items)
    stage2_biz = compute_business_metrics(pure_stage2_recs, test, k)

    # Stage 3 Re-Ranked (XGBoost + MMR + Category Cap)
    stage3_acc = evaluate_recommendations(v4_reranked_recs, test_user_items, k, total_items)
    stage3_div = evaluate_diversity_metrics(v4_reranked_recs, category_map, similarity_dict, total_items)
    stage3_biz = compute_business_metrics(v4_reranked_recs, test, k)

    # 6. Log to MLflow under recsys_v4
    print("\n[6/6] Logging V4 Re-Ranking Experiment to MLflow...")
    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment("recsys_v4")

        with mlflow.start_run(run_name="v4_reranking_diversity_benchmark"):
            mlflow.log_param("stage_1_retrieval_pool", 150)
            mlflow.log_param("stage_2_ranking_pool", 30)
            mlflow.log_param("stage_3_final_k", k)
            mlflow.log_param("mmr_lambda", 0.7)
            mlflow.log_param("max_per_category", 2)

            for prefix, acc_dict, div_dict, biz_dict in [
                ("pop", pop_acc, pop_div, pop_biz),
                ("stage2_pure", stage2_acc, stage2_div, stage2_biz),
                ("stage3_reranked", stage3_acc, stage3_div, stage3_biz),
            ]:
                for key, val in {**acc_dict, **div_dict, **biz_dict}.items():
                    if isinstance(val, (int, float)):
                        clean_key = f"{prefix}_{key}".replace("@", "_at_")
                        mlflow.log_metric(clean_key, val)

        print("  V4 benchmark logged to MLflow under 'recsys_v4'!")
    except Exception as e:
        print(f"  Warning: MLflow logging skipped ({e})")

    # Print Full Comparative Scorecard
    elapsed = time.time() - start_time
    print("\n" + "=" * 90)
    print("V4 SCORECARD: STAGE 2 PURE RANKER vs. STAGE 3 DIVERSIFIED RE-RANKER")
    print("=" * 90)
    header = f"{'Metric':<25} {'Popularity':<14} {'Stage 2 (Pure)':<16} {'Stage 3 (Re-Ranked)':<20} {'Re-Rank Lift':<14}"
    print(header)
    print("-" * 90)

    # Accuracy Metrics
    for m in [f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"hit_rate@{k}", f"mrr@{k}"]:
        vp = pop_acc[m]
        v2 = stage2_acc[m]
        v3 = stage3_acc[m]
        lift = ((v3 - v2) / v2) * 100 if v2 > 0 else 0.0
        print(f"{m:<25} {vp:<14.6f} {v2:<16.6f} {v3:<20.6f} {lift:+.1f}%")

    print("-" * 90)
    print("DIVERSITY & CATALOG SPREAD METRICS:")
    for d, label in [
        ("intra_list_diversity", "Intra-List Diversity (ILD)"),
        ("category_entropy", "Category Shannon Entropy"),
        ("unique_items_recommended", "Unique Items Surfaced"),
        ("catalog_coverage", "Catalog Coverage (%)"),
    ]:
        vp = pop_div[d]
        v2 = stage2_div[d]
        v3 = stage3_div[d]
        lift = ((v3 - v2) / v2) * 100 if v2 > 0 else 0.0
        if isinstance(v2, int):
            print(f"{label:<25} {vp:<14} {v2:<16} {v3:<20} {lift:+.1f}%")
        else:
            print(f"{label:<25} {vp:<14.4f} {v2:<16.4f} {v3:<20.4f} {lift:+.1f}%")

    print("-" * 90)
    print("BUSINESS CONVERSION METRICS:")
    for b in ["view_rate", "cart_rate", "purchase_rate"]:
        vp = pop_biz[b]
        v2 = stage2_biz[b]
        v3 = stage3_biz[b]
        lift = ((v3 - v2) / v2) * 100 if v2 > 0 else 0.0
        print(f"{b:<25} {vp:<14.6f} {v2:<16.6f} {v3:<20.6f} {lift:+.1f}%")

    print("=" * 90)
    print(f"Total V4 pipeline execution time: {elapsed:.1f} seconds")
    print("=" * 90)


if __name__ == "__main__":
    main()
