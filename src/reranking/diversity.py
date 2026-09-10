"""
Re-Ranking & Diversity Layer (Stage 3)
=====================================
Implements post-ranking business rules, diversity optimization,
and catalog exploration to prevent filter bubbles and popularity bias.

Techniques:
    1. Maximal Marginal Relevance (MMR)
    2. Category Capping (e.g. max 2 items per category)
    3. Freshness / Exploration Re-weighting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set


def compute_item_similarity_fast(
    item_a: int,
    item_b: int,
    similarity_map: dict,
    category_map: dict,
) -> float:
    """
    Compute pairwise similarity between item_a and item_b.
    Combines Item-CF cosine similarity and category match.
    """
    if item_a == item_b:
        return 1.0

    # Check Item-CF similarity if available
    sim_cf = 0.0
    if item_a in similarity_map:
        for sim_item, score in similarity_map[item_a]:
            if sim_item == item_b:
                sim_cf = float(score)
                break

    # Check category match safely with pd.NA handling
    cat_a = category_map.get(item_a, None)
    cat_b = category_map.get(item_b, None)
    cat_sim = 0.0
    if pd.notna(cat_a) and pd.notna(cat_b) and (cat_a == cat_b) is True:
        cat_sim = 1.0

    # Blended similarity
    return 0.6 * sim_cf + 0.4 * cat_sim


def rerank_user_mmr(
    ranked_candidates: List[dict],
    k: int = 10,
    lambda_param: float = 0.7,
    max_per_category: int = 2,
    similarity_map: dict = None,
    category_map: dict = None,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) re-ranking with category capping.
    """
    if not ranked_candidates:
        return []

    if similarity_map is None:
        similarity_map = {}
    if category_map is None:
        category_map = {}

    selected_items: List[int] = []
    category_counts: Dict[object, int] = {}
    candidate_pool = list(ranked_candidates)

    # Normalize scores to [0, 1] range for fair comparison with similarity
    scores = [c["score"] for c in candidate_pool]
    min_s = min(scores)
    max_s = max(scores)
    score_range = max(max_s - min_s, 1e-6)

    for c in candidate_pool:
        c["norm_score"] = (c["score"] - min_s) / score_range

    while len(selected_items) < k and candidate_pool:
        best_idx = -1
        best_mmr_val = -float("inf")

        for idx, cand in enumerate(candidate_pool):
            item_id = cand["itemid"]
            cat_id = category_map.get(item_id, None)

            # Check category cap constraint
            if pd.notna(cat_id) and category_counts.get(cat_id, 0) >= max_per_category:
                continue

            rel_score = cand["norm_score"]

            if not selected_items:
                mmr_val = rel_score
            else:
                max_sim = max(
                    compute_item_similarity_fast(item_id, sel_id, similarity_map, category_map)
                    for sel_id in selected_items
                )
                mmr_val = lambda_param * rel_score - (1.0 - lambda_param) * max_sim

            if mmr_val > best_mmr_val:
                best_mmr_val = mmr_val
                best_idx = idx

        if best_idx == -1:
            # Relax category quota for remaining slots
            remaining = [c["itemid"] for c in candidate_pool if c["itemid"] not in selected_items]
            selected_items.extend(remaining[: k - len(selected_items)])
            break

        chosen = candidate_pool.pop(best_idx)
        chosen_id = chosen["itemid"]
        selected_items.append(chosen_id)

        chosen_cat = category_map.get(chosen_id, None)
        if pd.notna(chosen_cat):
            category_counts[chosen_cat] = category_counts.get(chosen_cat, 0) + 1

    return selected_items


def rerank_batch(
    user_ranked_candidates: Dict[int, List[dict]],
    k: int = 10,
    lambda_param: float = 0.7,
    max_per_category: int = 2,
    similarity_map: dict = None,
    category_map: dict = None,
) -> Dict[int, List[int]]:
    """
    Apply MMR and category capping across all users in batch.
    """
    reranked_results = {}
    for user_id, cand_list in user_ranked_candidates.items():
        reranked_results[user_id] = rerank_user_mmr(
            cand_list,
            k=k,
            lambda_param=lambda_param,
            max_per_category=max_per_category,
            similarity_map=similarity_map,
            category_map=category_map,
        )
    return reranked_results
