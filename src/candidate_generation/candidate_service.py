"""
Candidate Service
=================
Combines candidates from multiple sources into a unified candidate pool.
Sources: Popularity, Item-CF, User History.
"""

import pandas as pd
from pathlib import Path
import joblib

from src.candidate_generation.popularity import get_popular_candidates
from src.candidate_generation.item_cf import get_cf_candidates


def generate_candidates_for_user(
    user_id: int,
    user_history: list,
    similarity_dict: dict,
    popular_items: list,
    config: dict,
) -> pd.DataFrame:
    """
    Generate a candidate pool for a single user by combining multiple sources.
    
    Sources:
        1. Item-CF: similar items to the user's history
        2. Popularity: globally popular items (always included as fallback)
    
    Each candidate is tagged with its source for feature engineering.
    
    Returns DataFrame with columns: [itemid, candidate_source]
    """
    pool_size = config["candidate_generation"]["pool_size"]
    cf_top_n = config["candidate_generation"]["item_cf_top_n"]

    candidates = {}

    # Source 1: Item-CF candidates
    cf_items = get_cf_candidates(user_history, similarity_dict, top_n=cf_top_n)
    for item in cf_items:
        candidates[item] = "item_cf"

    # Source 2: Popularity fallback (fill remaining slots)
    for item in popular_items:
        if item not in candidates and item not in set(user_history):
            candidates[item] = "popularity"
        if len(candidates) >= pool_size:
            break

    result = pd.DataFrame(
        [
            {"itemid": item_id, "candidate_source": source}
            for item_id, source in candidates.items()
        ]
    )

    return result.head(pool_size)


def generate_candidates_batch(
    test_users: list,
    train: pd.DataFrame,
    similarity_dict: dict,
    popular_items: list,
    config: dict,
) -> pd.DataFrame:
    """
    Generate candidate pools for a batch of test users.
    
    Returns DataFrame with columns: [visitorid, itemid, candidate_source]
    """
    print(f"Generating candidate pools for {len(test_users):,} users...")

    # Precompute user histories from training data
    user_histories = (
        train.groupby("visitorid")["itemid"].apply(list).to_dict()
    )

    all_candidates = []

    for i, user_id in enumerate(test_users):
        history = user_histories.get(user_id, [])
        candidates = generate_candidates_for_user(
            user_id, history, similarity_dict, popular_items, config
        )
        candidates["visitorid"] = user_id
        all_candidates.append(candidates)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1:,}/{len(test_users):,} users")

    result = pd.concat(all_candidates, ignore_index=True)

    print(f"  Total candidates generated: {len(result):,}")
    print(f"  Avg candidates/user: {len(result) / len(test_users):.1f}")
    print(f"  Source distribution:")
    print(result["candidate_source"].value_counts().to_string())

    return result
