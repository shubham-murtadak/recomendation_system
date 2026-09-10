"""
Interaction Features Module
===========================
Computes per (user, item) pair features from training data.
"""

import pandas as pd
from pathlib import Path


def build_interaction_features(train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute (user, item) pair features from training events.
    
    Features:
        - user_item_view_count: how many times this user viewed this item
        - user_item_cart_count: how many times added to cart
        - user_item_purchase_count: how many times purchased
        - user_item_total_score: sum of interaction weights for this pair
    """
    print("Building interaction features...")

    # Event type counts per (user, item)
    pair_events = (
        train.groupby(["visitorid", "itemid", "event"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["view", "addtocart", "transaction"]:
        if col not in pair_events.columns:
            pair_events[col] = 0

    pair_events = pair_events.rename(
        columns={
            "view": "user_item_view_count",
            "addtocart": "user_item_cart_count",
            "transaction": "user_item_purchase_count",
        }
    )

    # Total interaction score per pair
    pair_score = (
        train.groupby(["visitorid", "itemid"])["interaction_score"]
        .sum()
        .reset_index()
        .rename(columns={"interaction_score": "user_item_total_score"})
    )

    interaction_features = pair_events[
        [
            "visitorid",
            "itemid",
            "user_item_view_count",
            "user_item_cart_count",
            "user_item_purchase_count",
        ]
    ].merge(pair_score, on=["visitorid", "itemid"], how="left")

    print(f"  Built features for {len(interaction_features):,} (user, item) pairs")

    return interaction_features


def run_interaction_features(config: dict) -> pd.DataFrame:
    """Load train data and build interaction features."""
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed_dir / "train_events.parquet")
    interaction_features = build_interaction_features(train)

    output_path = features_dir / "interaction_features.parquet"
    interaction_features.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return interaction_features
