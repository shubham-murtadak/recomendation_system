"""
Item Features Module
====================
Computes per-item aggregations from training data for the ranking model.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_item_features(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute item-level features from training events.
    
    Features:
        - item_view_count: total views for this item
        - item_cart_count: total add-to-carts
        - item_purchase_count: total transactions
        - item_unique_visitors: distinct users who interacted
        - item_popularity_score: weighted score (1*views + 3*carts + 5*purchases)
        - days_since_item_last_interaction: recency from end of train period
        - item_category: category of the item
    """
    print("Building item features...")

    max_train_time = train["datetime"].max()
    weights = config["interaction_weights"]

    # Basic interaction counts per event type
    event_counts = (
        train.groupby(["itemid", "event"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["view", "addtocart", "transaction"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    event_counts = event_counts.rename(
        columns={
            "view": "item_view_count",
            "addtocart": "item_cart_count",
            "transaction": "item_purchase_count",
        }
    )
    event_counts = event_counts[
        ["itemid", "item_view_count", "item_cart_count", "item_purchase_count"]
    ]

    # Popularity score
    event_counts["item_popularity_score"] = (
        weights["view"] * event_counts["item_view_count"]
        + weights["addtocart"] * event_counts["item_cart_count"]
        + weights["transaction"] * event_counts["item_purchase_count"]
    )

    # Aggregations
    item_agg = train.groupby("itemid").agg(
        item_unique_visitors=("visitorid", "nunique"),
        item_last_interaction=("datetime", "max"),
    ).reset_index()

    # Recency
    item_agg["days_since_item_last_interaction"] = (
        (max_train_time - item_agg["item_last_interaction"]).dt.total_seconds() / 86400
    ).round(2)
    item_agg = item_agg.drop(columns=["item_last_interaction"])

    # Category (take the most recent category from training data)
    item_cat = (
        train[train["categoryid"].notna()]
        .sort_values("datetime")
        .drop_duplicates(subset="itemid", keep="last")[["itemid", "categoryid"]]
        .rename(columns={"categoryid": "item_category"})
    )

    # Merge all
    item_features = event_counts.merge(item_agg, on="itemid", how="left")
    item_features = item_features.merge(item_cat, on="itemid", how="left")

    print(f"  Built features for {len(item_features):,} items")
    print(f"  Columns: {list(item_features.columns)}")

    return item_features


def run_item_features(config: dict) -> pd.DataFrame:
    """Load train data and build item features."""
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed_dir / "train_events.parquet")
    item_features = build_item_features(train, config)

    output_path = features_dir / "item_features.parquet"
    item_features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return item_features
