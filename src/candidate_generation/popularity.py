"""
Popularity Baseline (Model 0)
=============================
The simplest recommender: recommend the most popular items to everyone.
Every sophisticated model must beat this baseline.
"""

import pandas as pd
from pathlib import Path


def build_popularity_ranking(train: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute global popularity ranking using weighted interaction scores.
    
    popularity_score = 1 * views + 3 * carts + 5 * purchases
    
    Returns a DataFrame sorted by popularity_score descending.
    """
    print("Building popularity baseline...")

    weights = config["interaction_weights"]

    # Count events per item
    item_events = (
        train.groupby(["itemid", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["view", "addtocart", "transaction"]:
        if col not in item_events.columns:
            item_events[col] = 0

    item_events["popularity_score"] = (
        weights["view"] * item_events["view"]
        + weights["addtocart"] * item_events["addtocart"]
        + weights["transaction"] * item_events["transaction"]
    )

    popularity = item_events[["itemid", "popularity_score"]].sort_values(
        "popularity_score", ascending=False
    ).reset_index(drop=True)

    popularity["popularity_rank"] = range(1, len(popularity) + 1)

    print(f"  Ranked {len(popularity):,} items by popularity")
    print(f"  Top 5 items:")
    print(popularity.head().to_string(index=False))

    return popularity


def get_popular_candidates(popularity: pd.DataFrame, n: int) -> list:
    """Return the top-N popular item IDs."""
    return popularity.head(n)["itemid"].tolist()


def recommend_popular(popularity: pd.DataFrame, k: int = 10) -> list:
    """
    Popularity baseline recommendation: same Top-K for every user.
    """
    return popularity.head(k)["itemid"].tolist()


def run_popularity(config: dict) -> pd.DataFrame:
    """Build and save the popularity ranking."""
    processed_dir = Path(config["data"]["processed_dir"])
    features_dir = Path(config["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(processed_dir / "train_events.parquet")
    popularity = build_popularity_ranking(train, config)

    output_path = features_dir / "popularity_ranking.parquet"
    popularity.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return popularity
