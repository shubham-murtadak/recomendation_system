"""
Diversity & Catalog Exploration Evaluation Metrics
==================================================
Measures Intra-List Diversity (ILD), Category Entropy,
and Catalog Spread for recommended item sets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter


def compute_intra_list_diversity(
    recommendations: Dict[int, List[int]],
    category_map: dict,
    similarity_map: dict = None,
) -> float:
    """
    Computes Intra-List Diversity (ILD):
        ILD = (1 / |U|) * sum_{u} [ (2 / (K*(K-1))) * sum_{i < j} (1 - sim(i, j)) ]
        
    Higher value (closer to 1.0) means recommended items are diverse and not clones.
    """
    if not recommendations:
        return 0.0

    if similarity_map is None:
        similarity_map = {}

    ild_scores = []

    for user_id, items in recommendations.items():
        n = len(items)
        if n <= 1:
            continue

        pair_dissimilarities = []
        for i in range(n):
            for j in range(i + 1, n):
                item_a = items[i]
                item_b = items[j]

                # Check category similarity safely
                cat_a = category_map.get(item_a, None)
                cat_b = category_map.get(item_b, None)
                cat_sim = 0.0
                if pd.notna(cat_a) and pd.notna(cat_b) and (cat_a == cat_b) is True:
                    cat_sim = 1.0

                # Check CF similarity
                sim_cf = 0.0
                if item_a in similarity_map:
                    for s_item, score in similarity_map[item_a]:
                        if s_item == item_b:
                            sim_cf = float(score)
                            break

                sim = 0.6 * sim_cf + 0.4 * cat_sim
                pair_dissimilarities.append(1.0 - sim)

        if pair_dissimilarities:
            ild_scores.append(np.mean(pair_dissimilarities))

    return float(np.mean(ild_scores)) if ild_scores else 0.0


def compute_category_entropy(
    recommendations: Dict[int, List[int]],
    category_map: dict,
) -> float:
    """
    Computes Shannon Entropy over recommended product categories.
    Higher entropy indicates broader, more uniform category exposure.
    """
    category_counts = Counter()
    total_recs = 0

    for user_id, items in recommendations.items():
        for item in items:
            cat = category_map.get(item, "unknown")
            cat_key = str(cat) if pd.notna(cat) else "unknown"
            category_counts[cat_key] += 1
            total_recs += 1

    if total_recs == 0:
        return 0.0

    entropy = 0.0
    for count in category_counts.values():
        p = count / total_recs
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)


def evaluate_diversity_metrics(
    recommendations: Dict[int, List[int]],
    category_map: dict,
    similarity_map: dict,
    total_catalog_items: int,
) -> dict:
    """
    Compute full suite of diversity, entropy, and catalog spread metrics.
    """
    all_recommended_items = set()
    for items in recommendations.values():
        all_recommended_items.update(items)

    ild = compute_intra_list_diversity(recommendations, category_map, similarity_map)
    entropy = compute_category_entropy(recommendations, category_map)
    unique_items = len(all_recommended_items)
    catalog_coverage = unique_items / max(total_catalog_items, 1)

    return {
        "intra_list_diversity": round(ild, 4),
        "category_entropy": round(entropy, 4),
        "unique_items_recommended": unique_items,
        "catalog_coverage": round(catalog_coverage, 6),
    }
