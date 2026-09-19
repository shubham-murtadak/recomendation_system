import pytest
from src.reranking.diversity import rerank_user_mmr
from src.candidate_generation.item_cf import get_cf_candidates


def test_mmr_reranking_basic():
    """Verify that MMR properly selects candidates while respecting category limits."""
    candidates = [
        {"itemid": 101, "score": 0.95},
        {"itemid": 102, "score": 0.92},
        {"itemid": 103, "score": 0.90},
        {"itemid": 201, "score": 0.85},
        {"itemid": 202, "score": 0.80},
        {"itemid": 301, "score": 0.75},
    ]

    similarity_map = {
        101: [(102, 0.9), (103, 0.85)],
        102: [(101, 0.9), (103, 0.8)],
        103: [(101, 0.85), (102, 0.8)],
        201: [(202, 0.8)],
        202: [(201, 0.8)],
        301: [],
    }

    category_map = {
        101: 1,
        102: 1,
        103: 1,
        201: 2,
        202: 2,
        301: 3,
    }

    # Test with max_per_category=2, k=4
    reranked_ids = rerank_user_mmr(
        ranked_candidates=candidates,
        k=4,
        lambda_param=0.7,
        max_per_category=2,
        similarity_map=similarity_map,
        category_map=category_map,
    )

    assert len(reranked_ids) == 4
    # Ensure category 1 has at most 2 items
    cat_1_count = sum(1 for item_id in reranked_ids if category_map[item_id] == 1)
    assert cat_1_count <= 2
    assert reranked_ids[0] == 101  # Top item should be preserved


def test_cf_candidate_retrieval_empty_history():
    """Verify fallback when user has no prior history."""
    sim_dict = {1: [(2, 0.8), (3, 0.5)]}
    candidates = get_cf_candidates(user_history=[], similarity_dict=sim_dict, top_n=5)
    assert candidates == []


def test_cf_candidate_retrieval():
    """Verify collaborative filtering retrieval on known seed."""
    sim_dict = {10: [(20, 0.85), (30, 0.75), (40, 0.60)]}
    candidates = get_cf_candidates(user_history=[10], similarity_dict=sim_dict, top_n=2)
    assert len(candidates) == 2
    assert candidates[0] == 20
    assert candidates[1] == 30
