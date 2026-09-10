"""
Model Pipeline Runner
=====================
Orchestrates the full model training and evaluation pipeline:
    Feature Engineering -> Candidate Generation -> Ranking -> Evaluation
    
Compares Popularity Baseline vs Personalized (Logistic Regression) model.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import mlflow

from src.data.ingestion import load_config
from src.features.user_features import build_user_features, run_user_features
from src.features.item_features import build_item_features, run_item_features
from src.features.interaction_features import build_interaction_features, run_interaction_features
from src.candidate_generation.popularity import build_popularity_ranking, get_popular_candidates, recommend_popular
from src.candidate_generation.item_cf import build_cooccurrence_matrix, compute_item_similarity, run_item_cf
from src.candidate_generation.candidate_service import generate_candidates_batch
from src.ranking.dataset import build_ranking_dataset, get_feature_columns
from src.ranking.train import train_ranking_model
from src.ranking.predict import predict_for_user
from src.evaluation.ranking_metrics import evaluate_recommendations
from src.evaluation.business_metrics import compute_business_metrics


def main():
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    k = config["evaluation"]["top_k"]

    start_time = time.time()

    # =========================================================
    # 1. Load processed data
    # =========================================================
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train = pd.read_parquet(processed_dir / "train_events.parquet")
    test = pd.read_parquet(processed_dir / "test_events.parquet")

    print(f"Train: {len(train):,} events | Test: {len(test):,} events")

    # =========================================================
    # 2. Feature Engineering
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: FEATURE ENGINEERING")
    print("=" * 60)

    user_features = build_user_features(train)
    item_features = build_item_features(train, config)
    interaction_features = build_interaction_features(train)

    # Save features
    features_dir.mkdir(parents=True, exist_ok=True)
    user_features.to_parquet(features_dir / "user_features.parquet", index=False)
    item_features.to_parquet(features_dir / "item_features.parquet", index=False)
    interaction_features.to_parquet(features_dir / "interaction_features.parquet", index=False)

    # =========================================================
    # 3. Popularity Baseline
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: POPULARITY BASELINE")
    print("=" * 60)

    popularity = build_popularity_ranking(train, config)
    popular_items = get_popular_candidates(
        popularity, config["candidate_generation"]["popularity_top_n"]
    )
    popularity_recs = recommend_popular(popularity, k=k)

    # =========================================================
    # 4. Item-CF
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: ITEM-ITEM COLLABORATIVE FILTERING")
    print("=" * 60)

    similarity_dict = run_item_cf(config)

    # =========================================================
    # 5. Candidate Generation for evaluation users
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: CANDIDATE GENERATION")
    print("=" * 60)

    # Only evaluate users who appear in BOTH train and test (not cold-start for V1)
    train_users = set(train["visitorid"].unique())
    test_users_all = set(test["visitorid"].unique())
    eval_users = list(train_users & test_users_all)

    # Sample for tractability (full evaluation on all 23K overlap users)
    max_eval_users = min(len(eval_users), 5000)
    np.random.seed(config["ranking"]["random_state"])
    eval_users_sample = list(np.random.choice(eval_users, max_eval_users, replace=False))

    print(f"Evaluating on {max_eval_users:,} users (out of {len(eval_users):,} overlap users)")

    candidates = generate_candidates_batch(
        eval_users_sample, train, similarity_dict, popular_items, config
    )

    # =========================================================
    # 6. Build Ranking Dataset & Train
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 4: RANKING MODEL")
    print("=" * 60)

    ranking_data = build_ranking_dataset(
        candidates, test, user_features, item_features, interaction_features, config
    )

    pipeline, feature_cols = train_ranking_model(ranking_data, config)

    # =========================================================
    # 7. Generate Recommendations & Evaluate
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 5: EVALUATION")
    print("=" * 60)

    # Build ground truth
    test_user_items = (
        test[test["visitorid"].isin(eval_users_sample)]
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    total_items = train["itemid"].nunique()

    # --- Evaluate POPULARITY BASELINE ---
    print("\n--- Popularity Baseline ---")
    pop_recommendations = {
        user_id: popularity_recs for user_id in eval_users_sample
    }

    pop_metrics = evaluate_recommendations(
        pop_recommendations, test_user_items, k, total_items
    )
    pop_biz = compute_business_metrics(pop_recommendations, test, k)

    print(f"  NDCG@{k}:     {pop_metrics[f'ndcg@{k}']:.6f}")
    print(f"  Recall@{k}:   {pop_metrics[f'recall@{k}']:.6f}")
    print(f"  Precision@{k}: {pop_metrics[f'precision@{k}']:.6f}")
    print(f"  Hit Rate@{k}: {pop_metrics[f'hit_rate@{k}']:.6f}")
    print(f"  MRR@{k}:      {pop_metrics[f'mrr@{k}']:.6f}")
    print(f"  Coverage:     {pop_metrics['coverage']:.6f}")
    print(f"  View Rate:    {pop_biz['view_rate']:.6f}")
    print(f"  Cart Rate:    {pop_biz['cart_rate']:.6f}")
    print(f"  Purchase Rate:{pop_biz['purchase_rate']:.6f}")

    # --- Evaluate PERSONALIZED MODEL ---
    print("\n--- Personalized (Logistic Regression) ---")
    personalized_recommendations = {}

    for user_id in eval_users_sample:
        result = predict_for_user(
            user_id, candidates, user_features, item_features,
            interaction_features, pipeline, feature_cols
        )
        personalized_recommendations[user_id] = result["itemid"].tolist()[:k]

    pers_metrics = evaluate_recommendations(
        personalized_recommendations, test_user_items, k, total_items
    )
    pers_biz = compute_business_metrics(personalized_recommendations, test, k)

    print(f"  NDCG@{k}:     {pers_metrics[f'ndcg@{k}']:.6f}")
    print(f"  Recall@{k}:   {pers_metrics[f'recall@{k}']:.6f}")
    print(f"  Precision@{k}: {pers_metrics[f'precision@{k}']:.6f}")
    print(f"  Hit Rate@{k}: {pers_metrics[f'hit_rate@{k}']:.6f}")
    print(f"  MRR@{k}:      {pers_metrics[f'mrr@{k}']:.6f}")
    print(f"  Coverage:     {pers_metrics['coverage']:.6f}")
    print(f"  View Rate:    {pers_biz['view_rate']:.6f}")
    print(f"  Cart Rate:    {pers_biz['cart_rate']:.6f}")
    print(f"  Purchase Rate:{pers_biz['purchase_rate']:.6f}")

    # --- Log to MLflow ---
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="v1_evaluation_comparison"):
        for name, metrics in [("pop", pop_metrics), ("pers", pers_metrics)]:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    clean_name = f"{name}_{metric_name}".replace("@", "_at_")
                    mlflow.log_metric(clean_name, value)

        for name, biz in [("pop", pop_biz), ("pers", pers_biz)]:
            for metric_name, value in biz.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{name}_{metric_name}", value)

    # --- Summary ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Popularity':<15} {'Personalized':<15} {'Winner':<12}")
    print("-" * 62)
    for metric in [f"ndcg@{k}", f"recall@{k}", f"precision@{k}", f"hit_rate@{k}", f"mrr@{k}", "coverage"]:
        p = pop_metrics[metric]
        r = pers_metrics[metric]
        winner = "Personalized" if r > p else ("Popularity" if p > r else "Tie")
        print(f"{metric:<20} {p:<15.6f} {r:<15.6f} {winner:<12}")

    print(f"\nTotal pipeline time: {elapsed:.1f} seconds")
    print("All metrics logged to MLflow.")


if __name__ == "__main__":
    main()
