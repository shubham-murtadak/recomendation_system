"""
Ranking Metrics Module
======================
Implements standard recommendation system evaluation metrics from scratch.
Primary: NDCG@K | Secondary: Recall@K
"""

import numpy as np


def precision_at_k(recommended: list, actual: set, k: int) -> float:
    """
    Of the top-K items we recommended, how many were relevant?
    """
    if k == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant = sum(1 for item in recommended_k if item in actual)
    return relevant / k


def recall_at_k(recommended: list, actual: set, k: int) -> float:
    """
    Of all relevant items, how many did we retrieve in top-K?
    """
    if len(actual) == 0:
        return 0.0
    recommended_k = recommended[:k]
    relevant = sum(1 for item in recommended_k if item in actual)
    return relevant / len(actual)


def ndcg_at_k(recommended: list, actual: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Rewards placing relevant items higher in the ranking.
    """
    recommended_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (all relevant items at the top)
    ideal_k = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(recommended: list, actual: set, k: int) -> float:
    """
    Binary: did we get at least one relevant item in top-K?
    """
    recommended_k = recommended[:k]
    return 1.0 if any(item in actual for item in recommended_k) else 0.0


def mrr_at_k(recommended: list, actual: set, k: int) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first relevant item in top-K.
    """
    recommended_k = recommended[:k]
    for i, item in enumerate(recommended_k):
        if item in actual:
            return 1.0 / (i + 1)
    return 0.0


def coverage(all_recommended_items: set, total_items: int) -> float:
    """
    Fraction of the total item catalog that appears in any recommendation.
    """
    if total_items == 0:
        return 0.0
    return len(all_recommended_items) / total_items


def evaluate_recommendations(
    user_recommendations: dict,
    user_actuals: dict,
    k: int,
    total_items: int,
) -> dict:
    """
    Evaluate a recommendation system over all users.
    
    Args:
        user_recommendations: {user_id: [recommended_item_ids]}
        user_actuals: {user_id: set(actual_item_ids)}
        k: number of recommendations to evaluate
        total_items: total unique items in catalog
    
    Returns:
        dict with aggregated metrics
    """
    all_precision = []
    all_recall = []
    all_ndcg = []
    all_hit = []
    all_mrr = []
    all_recommended = set()

    for user_id in user_recommendations:
        if user_id not in user_actuals or len(user_actuals[user_id]) == 0:
            continue

        rec = user_recommendations[user_id]
        actual = user_actuals[user_id]

        all_precision.append(precision_at_k(rec, actual, k))
        all_recall.append(recall_at_k(rec, actual, k))
        all_ndcg.append(ndcg_at_k(rec, actual, k))
        all_hit.append(hit_rate_at_k(rec, actual, k))
        all_mrr.append(mrr_at_k(rec, actual, k))

        all_recommended.update(rec[:k])

    n_users = len(all_precision)

    metrics = {
        f"precision@{k}": np.mean(all_precision) if all_precision else 0.0,
        f"recall@{k}": np.mean(all_recall) if all_recall else 0.0,
        f"ndcg@{k}": np.mean(all_ndcg) if all_ndcg else 0.0,
        f"hit_rate@{k}": np.mean(all_hit) if all_hit else 0.0,
        f"mrr@{k}": np.mean(all_mrr) if all_mrr else 0.0,
        "coverage": coverage(all_recommended, total_items),
        "evaluated_users": n_users,
    }

    return metrics
