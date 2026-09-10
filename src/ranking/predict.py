"""
Ranking Prediction
==================
Uses trained ranking models (Logistic Regression or LightGBM)
to score and rank candidate items for a user or in batch.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def predict_for_user(
    user_id: int,
    candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    pipeline,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Score and rank candidate items for a single user (used in API).
    Returns DataFrame sorted by predicted probability descending.
    """
    # Filter candidates for this user
    user_candidates = candidates[candidates["visitorid"] == user_id].copy()

    if user_candidates.empty:
        return pd.DataFrame(columns=["itemid", "score"])

    # One-hot encode candidate_source
    user_candidates = pd.get_dummies(
        user_candidates, columns=["candidate_source"], prefix="src"
    )

    # Join features
    user_candidates = user_candidates.merge(
        user_features, on="visitorid", how="left"
    )
    user_candidates = user_candidates.merge(
        item_features, on="itemid", how="left"
    )
    user_candidates = user_candidates.merge(
        interaction_features, on=["visitorid", "itemid"], how="left"
    )

    # Fill NaNs
    user_candidates = user_candidates.fillna(0)

    # Category match if feature exists
    if "category_match" in feature_cols and "user_top_category" in user_candidates.columns and "item_category" in user_candidates.columns:
        user_candidates["category_match"] = (
            user_candidates["user_top_category"] == user_candidates["item_category"]
        ).astype(int)

    # Ensure all feature columns exist in correct order
    for col in feature_cols:
        if col not in user_candidates.columns:
            user_candidates[col] = 0

    X = user_candidates[feature_cols].values

    # Predict probability of interaction
    scores = pipeline.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "itemid": user_candidates["itemid"].values,
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    return result


def predict_for_users_batch(
    eval_users: list,
    candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    pipeline,
    feature_cols: list,
    k: int = 10,
) -> dict:
    """
    High-performance vectorized batch prediction for thousands of evaluation users.
    Joins features across the candidate pool in bulk, scores in one forward pass,
    and extracts the top-K recommendations per user in seconds.
    """
    eval_set = set(eval_users)
    df = candidates[candidates["visitorid"].isin(eval_set)].copy()

    if df.empty:
        return {u: [] for u in eval_users}

    # One-hot encode candidate_source
    df = pd.get_dummies(df, columns=["candidate_source"], prefix="src")

    # Vectorized bulk merges
    df = df.merge(user_features, on="visitorid", how="left")
    df = df.merge(item_features, on="itemid", how="left")
    df = df.merge(interaction_features, on=["visitorid", "itemid"], how="left")

    df = df.fillna(0)

    # Category match if feature exists
    if "category_match" in feature_cols and "user_top_category" in df.columns and "item_category" in df.columns:
        df["category_match"] = (
            df["user_top_category"] == df["item_category"]
        ).astype(int)

    # Ensure all required features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values

    # Single-pass batch inference
    df["score"] = pipeline.predict_proba(X)[:, 1]

    # Sort descending by predicted probability within each user
    df_sorted = df.sort_values(["visitorid", "score"], ascending=[True, False])

    # Extract Top-K items per user
    top_k_map = (
        df_sorted.groupby("visitorid", observed=True)["itemid"]
        .apply(lambda items: items.head(k).tolist())
        .to_dict()
    )

    # Ensure every requested user has a recommendation list
    return {u: top_k_map.get(u, []) for u in eval_users}


def predict_ranked_candidates_batch(
    eval_users: list,
    candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    pipeline,
    feature_cols: list,
    top_n: int = 30,
) -> dict:
    """
    Scores candidates in vectorized batch and returns top-N candidate dictionaries
    with itemid and score for Stage 3 Re-Ranking.
    """
    eval_set = set(eval_users)
    df = candidates[candidates["visitorid"].isin(eval_set)].copy()

    if df.empty:
        return {u: [] for u in eval_users}

    df = pd.get_dummies(df, columns=["candidate_source"], prefix="src")
    df = df.merge(user_features, on="visitorid", how="left")
    df = df.merge(item_features, on="itemid", how="left")
    df = df.merge(interaction_features, on=["visitorid", "itemid"], how="left")
    df = df.fillna(0)

    if "category_match" in feature_cols and "user_top_category" in df.columns and "item_category" in df.columns:
        df["category_match"] = (
            df["user_top_category"] == df["item_category"]
        ).astype(int)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].values
    df["score"] = pipeline.predict_proba(X)[:, 1]

    # Sort descending by score within each user
    df_sorted = df.sort_values(["visitorid", "score"], ascending=[True, False])

    # Keep top_n candidates per user
    top_n_df = df_sorted.groupby("visitorid", observed=True).head(top_n)

    result = {}
    for uid, group in top_n_df.groupby("visitorid", observed=True):
        result[uid] = [
            {"itemid": int(row.itemid), "score": float(row.score)}
            for row in group.itertuples()
        ]

    for u in eval_users:
        if u not in result:
            result[u] = []

    return result


def load_ranking_model(model_name: str = "ranking_model.joblib") -> tuple:
    """Load the trained ranking model and feature columns."""
    models_dir = Path("models")
    pipeline = joblib.load(models_dir / model_name)
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")
    return pipeline, feature_cols
