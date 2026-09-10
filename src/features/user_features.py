"""
User Features Module
====================
Computes per-user aggregations from training data for the ranking model.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_user_features(train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-level features from training events.
    
    Features:
        - user_view_count: total views by this user
        - user_cart_count: total add-to-carts
        - user_purchase_count: total transactions
        - user_total_interactions: total events
        - user_unique_items: number of distinct items interacted with
        - user_unique_categories: number of distinct categories
        - user_avg_session_length: average events per session
        - days_since_user_last_interaction: recency from end of train period
        - user_top_category: most frequently interacted category
    """
    print("Building user features...")

    max_train_time = train["datetime"].max()

    # Basic interaction counts per event type
    event_counts = (
        train.groupby(["visitorid", "event"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all event columns exist
    for col in ["view", "addtocart", "transaction"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    event_counts = event_counts.rename(
        columns={
            "view": "user_view_count",
            "addtocart": "user_cart_count",
            "transaction": "user_purchase_count",
        }
    )
    event_counts = event_counts[
        ["visitorid", "user_view_count", "user_cart_count", "user_purchase_count"]
    ]

    # Aggregations
    user_agg = train.groupby("visitorid").agg(
        user_total_interactions=("itemid", "count"),
        user_unique_items=("itemid", "nunique"),
        user_unique_categories=("categoryid", "nunique"),
        user_last_interaction=("datetime", "max"),
        user_total_score=("interaction_score", "sum"),
    ).reset_index()

    # Recency
    user_agg["days_since_user_last_interaction"] = (
        (max_train_time - user_agg["user_last_interaction"]).dt.total_seconds() / 86400
    ).round(2)
    user_agg = user_agg.drop(columns=["user_last_interaction"])

    # Average session length
    session_lengths = train.groupby(["visitorid", "session_id"]).size().reset_index(name="sess_len")
    avg_session = session_lengths.groupby("visitorid")["sess_len"].mean().reset_index()
    avg_session = avg_session.rename(columns={"sess_len": "user_avg_session_length"})

    # Top category per user (mode)
    user_top_cat = (
        train[train["categoryid"].notna()]
        .groupby("visitorid")["categoryid"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        .reset_index()
        .rename(columns={"categoryid": "user_top_category"})
    )

    # Merge all
    user_features = event_counts.merge(user_agg, on="visitorid", how="left")
    user_features = user_features.merge(avg_session, on="visitorid", how="left")
    user_features = user_features.merge(user_top_cat, on="visitorid", how="left")

    print(f"  Built features for {len(user_features):,} users")
    print(f"  Columns: {list(user_features.columns)}")

    return user_features


def run_user_features(config: dict) -> pd.DataFrame:
    """Load train data and build user features."""
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed_dir / "train_events.parquet")
    user_features = build_user_features(train)

    output_path = features_dir / "user_features.parquet"
    user_features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return user_features
