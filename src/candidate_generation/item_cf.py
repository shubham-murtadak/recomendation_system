"""
Item-Item Collaborative Filtering
==================================
Finds similar items based on co-occurrence in user interaction histories.
If users who interacted with item A also interacted with item B,
then A and B are considered similar.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib


def build_cooccurrence_matrix(train: pd.DataFrame) -> tuple:
    """
    Build a user-item interaction matrix and compute item-item
    cosine similarity from co-occurrence patterns.
    
    Returns:
        (similarity_dict, item_id_map)
        
        similarity_dict: {item_id: [(similar_item, score), ...]}
        item_id_map: mapping from matrix index to original item_id
    """
    print("Building item-item co-occurrence matrix...")

    # Create user-item interaction matrix (binary: interacted or not)
    users = train["visitorid"].unique()
    items = train["itemid"].unique()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    # Build sparse matrix
    row_indices = train["visitorid"].map(user_to_idx).values
    col_indices = train["itemid"].map(item_to_idx).values
    data = np.ones(len(train))

    user_item_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(users), len(items)),
    )

    # Binarize (in case of multiple interactions)
    user_item_matrix = (user_item_matrix > 0).astype(np.float32)

    print(f"  Matrix shape: {user_item_matrix.shape}")
    print(f"  Sparsity: {1 - user_item_matrix.nnz / np.prod(user_item_matrix.shape):.6f}")

    return user_item_matrix, item_to_idx, idx_to_item


def compute_item_similarity(
    user_item_matrix: csr_matrix,
    idx_to_item: dict,
    top_n: int = 20,
    batch_size: int = 5000,
) -> dict:
    """
    Compute item-item cosine similarity in batches to avoid OOM.
    
    Returns dict mapping each item_id to its top_n most similar items.
    """
    print(f"  Computing item-item cosine similarity (top {top_n} per item)...")

    n_items = user_item_matrix.shape[1]
    item_matrix = user_item_matrix.T  # items x users

    similarity_dict = {}

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = item_matrix[start:end]

        # Compute similarity of this batch against all items
        sim_batch = cosine_similarity(batch, item_matrix)

        for i in range(sim_batch.shape[0]):
            item_idx = start + i
            item_id = idx_to_item[item_idx]

            # Get top_n similar items (exclude self)
            scores = sim_batch[i]
            scores[item_idx] = -1  # Exclude self

            top_indices = np.argpartition(scores, -top_n)[-top_n:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            similar_items = [
                (idx_to_item[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]
            similarity_dict[item_id] = similar_items

        if (end % 50000 == 0) or (end == n_items):
            print(f"    Processed {end:,}/{n_items:,} items")

    return similarity_dict


def get_cf_candidates(
    user_history: list, similarity_dict: dict, top_n: int = 20
) -> list:
    """
    Given a user's interaction history (list of item_ids),
    return candidate items from item-CF.
    
    For each item the user interacted with, retrieve similar items.
    Aggregate scores and deduplicate.
    """
    candidate_scores = {}

    for item_id in user_history:
        if item_id in similarity_dict:
            for similar_item, score in similarity_dict[item_id]:
                if similar_item not in set(user_history):
                    if similar_item in candidate_scores:
                        candidate_scores[similar_item] = max(
                            candidate_scores[similar_item], score
                        )
                    else:
                        candidate_scores[similar_item] = score

    # Sort by score and return top_n
    sorted_candidates = sorted(
        candidate_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    return [item_id for item_id, _ in sorted_candidates]


def run_item_cf(config: dict, force_recompute: bool = False) -> dict:
    """Build and save the item-item similarity model (with caching)."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "item_similarity.joblib"

    if output_path.exists() and not force_recompute:
        print(f"  Loading cached similarity model from {output_path}...")
        return joblib.load(output_path)

    processed_dir = Path(config["data"]["processed_dir"])
    train = pd.read_parquet(processed_dir / "train_events.parquet")

    top_n = config["candidate_generation"]["item_cf_top_n"]

    user_item_matrix, item_to_idx, idx_to_item = build_cooccurrence_matrix(train)
    similarity_dict = compute_item_similarity(
        user_item_matrix, idx_to_item, top_n=top_n
    )

    joblib.dump(similarity_dict, output_path)
    print(f"  Saved similarity model to {output_path}")

    return similarity_dict
