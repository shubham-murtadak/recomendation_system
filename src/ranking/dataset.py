"""
Ranking Dataset Builder
=======================
Creates the labeled dataset for the ranking model.
For each user, generates positive and negative examples
from their candidate pool.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_ranking_dataset(
    candidates: pd.DataFrame,
    test: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    interaction_features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Build labeled ranking dataset.
    
    Label logic:
        y = 1 if the user actually interacted with this item in the test period
        y = 0 otherwise (negative sample)
    
    Feature vector = user_features + item_features + interaction_features + candidate_source
    """
    print("Building ranking dataset...")

    neg_ratio = config["ranking"]["negative_sample_ratio"]

    # Ground truth: items each user interacted with in test
    test_interactions = set(
        zip(test["visitorid"].values, test["itemid"].values)
    )

    # Label candidates
    candidates["label"] = candidates.apply(
        lambda row: 1 if (row["visitorid"], row["itemid"]) in test_interactions else 0,
        axis=1,
    )

    n_positive = candidates["label"].sum()
    n_negative = (candidates["label"] == 0).sum()
    print(f"  Before sampling: {n_positive:,} positives, {n_negative:,} negatives")

    # Negative sampling: keep all positives, downsample negatives
    positives = candidates[candidates["label"] == 1]
    negatives = candidates[candidates["label"] == 0]

    target_neg = min(len(negatives), n_positive * neg_ratio)
    if target_neg > 0 and len(negatives) > 0:
        negatives_sampled = negatives.sample(
            n=target_neg, random_state=config["ranking"]["random_state"]
        )
    else:
        negatives_sampled = negatives

    dataset = pd.concat([positives, negatives_sampled], ignore_index=True)

    print(f"  After sampling: {len(positives):,} positives, {len(negatives_sampled):,} negatives")

    # Merge features
    print("  Joining features...")
    dataset = dataset.merge(user_features, on="visitorid", how="left")
    dataset = dataset.merge(item_features, on="itemid", how="left")
    dataset = dataset.merge(
        interaction_features, on=["visitorid", "itemid"], how="left"
    )

    # Fill NaN interaction features with 0 (no prior interaction)
    interaction_cols = [
        "user_item_view_count",
        "user_item_cart_count",
        "user_item_purchase_count",
        "user_item_total_score",
    ]
    for col in interaction_cols:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0)

    # One-hot encode candidate_source
    dataset = pd.get_dummies(dataset, columns=["candidate_source"], prefix="src")

    # Fill remaining NaNs
    dataset = dataset.fillna(0)

    print(f"  Final dataset shape: {dataset.shape}")
    print(f"  Label distribution:")
    print(dataset["label"].value_counts().to_string())

    return dataset


def get_feature_columns(dataset: pd.DataFrame) -> list:
    """Return list of feature columns (everything except IDs and label)."""
    exclude = {"visitorid", "itemid", "label", "user_top_category", "item_category"}
    return [c for c in dataset.columns if c not in exclude]
