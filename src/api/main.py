"""
FastAPI Recommendation Service
==============================
Provides high-performance REST API endpoints with interactive Swagger UI (/docs)
for real-time e-commerce recommendations.

Stages:
    - Stage 1: Candidate Retrieval (Item-CF + Popularity)
    - Stage 2: ML Ranking (XGBoost GBDT, DeepFM, or Logistic Regression)
    - Stage 3: Re-Ranking (Maximal Marginal Relevance + Category Capping)
"""

# Windows OpenMP safety import
import torch

import time
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException, Path as FPath
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

from src.candidate_generation.item_cf import get_cf_candidates
from src.reranking.diversity import rerank_user_mmr


# Pydantic Schemas for Swagger documentation
class RecommendationItem(BaseModel):
    rank: int = Field(..., description="1-based position in recommendation carousel")
    item_id: int = Field(..., description="Unique product ID")
    score: float = Field(..., description="Predicted interaction probability or relevance score")
    category_id: Optional[int] = Field(None, description="Product category ID")
    candidate_source: str = Field(..., description="Source: 'item_cf' or 'popularity'")


class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="Requested visitor ID")
    model_version: str = Field(..., description="Active model: v4 (Re-ranked), v2 (XGBoost), v3 (DeepFM), v1 (LR), popularity")
    is_cold_start: bool = Field(..., description="True if user has no prior history (cold-start fallback to popularity)")
    user_history_count: int = Field(..., description="Number of items previously interacted with by user")
    recommendations: List[RecommendationItem] = Field(..., description="Top-K personalized items")
    latency_ms: float = Field(..., description="Total server processing time in milliseconds")


class SimilarItem(BaseModel):
    rank: int = Field(..., description="1-based similarity rank")
    item_id: int = Field(..., description="Similar product ID")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    category_id: Optional[int] = Field(None, description="Category of similar item")


class SimilarItemsResponse(BaseModel):
    seed_item_id: int = Field(..., description="Target item ID")
    seed_category_id: Optional[int] = Field(None, description="Target item's category")
    similar_items: List[SimilarItem] = Field(..., description="Top similar items via Collaborative Filtering")
    count: int


class PopularItem(BaseModel):
    rank: int
    item_id: int
    popularity_score: float
    category_id: Optional[int]


class HealthResponse(BaseModel):
    status: str
    active_models: List[str]
    total_catalog_items: int
    total_users_in_store: int
    uptime_seconds: float


# Global Model Cache
state: Dict[str, Any] = {}
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models, feature stores, and lookups into memory on startup."""
    print("Initializing Recommendation Service & loading model artifacts...")
    models_dir = Path("models")
    features_dir = Path("data/features")

    # 1. Feature columns
    state["feature_cols"] = joblib.load(models_dir / "feature_columns.joblib")

    # 2. Ranking Models
    state["models"] = {}
    if (models_dir / "ranking_model_xgboost.joblib").exists():
        state["models"]["v2"] = joblib.load(models_dir / "ranking_model_xgboost.joblib")
        state["models"]["v4"] = state["models"]["v2"]  # V4 uses XGBoost + Stage 3 Re-Ranking
    elif (models_dir / "ranking_model.joblib").exists():
        state["models"]["v2"] = joblib.load(models_dir / "ranking_model.joblib")
        state["models"]["v4"] = state["models"]["v2"]

    if (models_dir / "ranking_model_logistic_regression.joblib").exists():
        state["models"]["v1"] = joblib.load(models_dir / "ranking_model_logistic_regression.joblib")

    if (models_dir / "ranking_model_deepfm.joblib").exists():
        state["models"]["v3"] = joblib.load(models_dir / "ranking_model_deepfm.joblib")

    # 3. Collaborative Filtering & Popularity
    state["similarity_dict"] = joblib.load(models_dir / "item_similarity.joblib")
    state["popular_items"] = joblib.load(models_dir / "popular_items.joblib")
    state["user_history"] = joblib.load(models_dir / "user_history.joblib")

    # 4. Feature Stores (Indexed for fast O(1) lookups)
    print("  Loading feature store indexes...")
    item_df = pd.read_parquet(features_dir / "item_features.parquet")
    state["item_features_indexed"] = item_df.set_index("itemid")
    state["category_map"] = dict(zip(item_df["itemid"], item_df["item_category"]))
    state["item_popularity_scores"] = dict(zip(item_df["itemid"], item_df["item_popularity_score"]))

    user_df = pd.read_parquet(features_dir / "user_features.parquet")
    state["user_features_indexed"] = user_df.set_index("visitorid")

    # Interaction feature lookup: (visitorid, itemid) -> row dict
    inter_df = pd.read_parquet(features_dir / "interaction_features.parquet")
    state["interaction_map"] = {
        (row.visitorid, row.itemid): {
            "user_item_view_count": row.user_item_view_count,
            "user_item_cart_count": row.user_item_cart_count,
            "user_item_purchase_count": row.user_item_purchase_count,
            "user_item_total_score": row.user_item_total_score,
        }
        for row in inter_df.itertuples()
    }

    print("Recommendation Service initialized successfully!")
    yield
    print("Shutting down Recommendation Service...")


app = FastAPI(
    title="Production E-Commerce Recommendation API",
    description="""
### Real-Time Multi-Stage Recommendation Engine (RetailRocket E-Commerce)

This API powers personalized product discovery through a production **3-stage funnel**:
* **Stage 1: Candidate Retrieval** — High-recall hybrid retrieval using Item-Item Collaborative Filtering (Cosine Matrix) + Popularity fallback (~100 items).
* **Stage 2: Precision Ranking** — Scores candidates using our benchmark champion **XGBoost GBDT**, **PyTorch DeepFM**, or **Logistic Regression**.
* **Stage 3: Re-Ranking Layer (V4)** — Balances accuracy with diversity using **Maximal Marginal Relevance (MMR)** and **Category Capping** constraints (max 2 items/category).

Explore and test the endpoints below with interactive **Swagger UI**!
""",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Service health and loaded model status."""
    return HealthResponse(
        status="healthy",
        active_models=list(state["models"].keys()) + ["popularity"],
        total_catalog_items=len(state["item_features_indexed"]),
        total_users_in_store=len(state["user_features_indexed"]),
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/popular", response_model=List[PopularItem], tags=["Discovery"])
def get_popular_products(
    k: int = Query(10, ge=1, le=100, description="Number of popular items to return")
):
    """Retrieve globally popular items based on weighted interaction scores."""
    pop_items = state["popular_items"][:k]
    results = []
    for rank, it in enumerate(pop_items, 1):
        results.append(
            PopularItem(
                rank=rank,
                item_id=int(it),
                popularity_score=float(state["item_popularity_scores"].get(it, 0.0)),
                category_id=int(state["category_map"].get(it)) if pd.notna(state["category_map"].get(it)) else None,
            )
        )
    return results


@app.get("/similar/{item_id}", response_model=SimilarItemsResponse, tags=["Collaborative Filtering"])
def get_similar_items(
    item_id: int = FPath(..., description="Target item ID to find similar items for"),
    k: int = Query(10, ge=1, le=50, description="Number of similar items to return"),
):
    """
    Retrieve products frequently co-browsed or purchased together
    computed via Item-Item Cosine Similarity matrix.
    """
    sim_dict = state["similarity_dict"]
    cat_map = state["category_map"]

    if item_id not in sim_dict:
        # Fallback to popular items from same category or overall
        seed_cat = cat_map.get(item_id, None)
        similar_items = [
            SimilarItem(
                rank=r,
                item_id=int(it),
                similarity_score=0.1,
                category_id=int(cat_map.get(it)) if pd.notna(cat_map.get(it)) else None,
            )
            for r, it in enumerate(state["popular_items"][:k], 1)
        ]
        return SimilarItemsResponse(
            seed_item_id=item_id,
            seed_category_id=int(seed_cat) if pd.notna(seed_cat) else None,
            similar_items=similar_items,
            count=len(similar_items),
        )

    pairs = sim_dict[item_id][:k]
    similar_items = []
    for r, (sim_id, score) in enumerate(pairs, 1):
        similar_items.append(
            SimilarItem(
                rank=r,
                item_id=int(sim_id),
                similarity_score=round(float(score), 4),
                category_id=int(cat_map.get(sim_id)) if pd.notna(cat_map.get(sim_id)) else None,
            )
        )

    seed_cat = cat_map.get(item_id, None)
    return SimilarItemsResponse(
        seed_item_id=item_id,
        seed_category_id=int(seed_cat) if pd.notna(seed_cat) else None,
        similar_items=similar_items,
        count=len(similar_items),
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(
    user_id: int = FPath(..., description="User ID / Visitor ID"),
    k: int = Query(10, ge=1, le=50, description="Number of items to recommend"),
    model_version: str = Query(
        "v4",
        regex="^(v4|v2|v3|v1|popularity)$",
        description="Model architecture version: v4 (Two-Stage + MMR Re-ranking), v2 (XGBoost GBDT), v3 (DeepFM), v1 (Logistic Regression), popularity",
    ),
    enable_reranking: bool = Query(
        True,
        description="Apply Stage 3 MMR diversity & category capping (used for v4)",
    ),
    max_per_category: int = Query(
        2,
        ge=1,
        le=5,
        description="Maximum items allowed from the exact same category in the top-K carousel",
    ),
    lambda_param: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="MMR diversity trade-off: 1.0 = Pure relevance, 0.0 = Pure diversity",
    ),
):
    """
    Generate personalized top-K recommendations for a user.

    - **Cold-Start Handling**: If the user is new / has no history, automatically serves top globally popular products.
    - **V4 Architecture**: Retrieves ~100 candidate items (Item-CF + Popularity) &rarr; Ranks candidates with XGBoost &rarr; Diversifies top candidates using MMR and category quota constraints.
    """
    t0 = time.time()
    user_history = state["user_history"].get(user_id, [])
    is_cold_start = len(user_history) == 0

    # 1. Cold-Start or Popularity Request
    if is_cold_start or model_version == "popularity":
        pop_slice = [it for it in state["popular_items"] if it not in set(user_history)][:k]
        recs = [
            RecommendationItem(
                rank=r,
                item_id=int(it),
                score=round(float(state["item_popularity_scores"].get(it, 0.0)), 4),
                category_id=int(state["category_map"].get(it)) if pd.notna(state["category_map"].get(it)) else None,
                candidate_source="popularity",
            )
            for r, it in enumerate(pop_slice, 1)
        ]
        return RecommendationResponse(
            user_id=user_id,
            model_version=model_version,
            is_cold_start=is_cold_start,
            user_history_count=len(user_history),
            recommendations=recs,
            latency_ms=round((time.time() - t0) * 1000, 2),
        )

    # 2. Stage 1: Candidate Retrieval (Hybrid: Item-CF + Popularity)
    cf_candidates = get_cf_candidates(user_history, state["similarity_dict"], top_n=20)
    candidate_dict = {}
    for it in cf_candidates:
        if it not in set(user_history):
            candidate_dict[it] = "item_cf"

    for it in state["popular_items"]:
        if it not in candidate_dict and it not in set(user_history):
            candidate_dict[it] = "popularity"
        if len(candidate_dict) >= 100:
            break

    candidate_items = list(candidate_dict.keys())
    candidate_sources = list(candidate_dict.values())

    # 3. Stage 2: Feature Assembly & Model Scoring
    # Fast O(1) user profile lookup
    if user_id in state["user_features_indexed"].index:
        user_row = state["user_features_indexed"].loc[user_id]
        user_dict = user_row.to_dict()
    else:
        user_dict = {}

    # Build feature rows
    feature_rows = []
    feature_cols = state["feature_cols"]

    for item_id, src in zip(candidate_items, candidate_sources):
        row = {"visitorid": user_id, "itemid": item_id}
        # Source one-hot
        row["src_popularity"] = 1 if src == "popularity" else 0
        row["src_item_cf"] = 1 if src == "item_cf" else 0

        # User features
        row.update(user_dict)

        # Item features
        if item_id in state["item_features_indexed"].index:
            item_row = state["item_features_indexed"].loc[item_id]
            row.update(item_row.to_dict())

        # Interaction features
        inter_row = state["interaction_map"].get((user_id, item_id), {})
        row.update(inter_row)

        # Category match
        u_top_cat = user_dict.get("user_top_category", None)
        i_cat = row.get("item_category", None)
        if pd.notna(u_top_cat) and pd.notna(i_cat) and (u_top_cat == i_cat):
            row["category_match"] = 1
        else:
            row["category_match"] = 0

        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows).fillna(0)
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0

    X = np.ascontiguousarray(feat_df[feature_cols].values, dtype=np.float32)

    # Choose model
    model = state["models"].get(model_version, state["models"].get("v2"))
    scores = model.predict_proba(X)[:, 1]

    # Assemble ranked candidates
    ranked_list = [
        {"itemid": int(it), "score": float(sc), "source": src}
        for it, sc, src in zip(candidate_items, scores, candidate_sources)
    ]
    ranked_list.sort(key=lambda x: x["score"], reverse=True)

    # 4. Stage 3: Re-Ranking Layer (if V4 or enable_reranking=True)
    if model_version == "v4" and enable_reranking:
        # Take top 30 candidates into Stage 3 MMR re-ranking
        top30 = ranked_list[:30]
        final_item_ids = rerank_user_mmr(
            top30,
            k=k,
            lambda_param=lambda_param,
            max_per_category=max_per_category,
            similarity_map=state["similarity_dict"],
            category_map=state["category_map"],
        )
        item_score_map = {c["itemid"]: c["score"] for c in top30}
        item_src_map = {c["itemid"]: c["source"] for c in top30}

        recs = [
            RecommendationItem(
                rank=r,
                item_id=it,
                score=round(float(item_score_map.get(it, 0.0)), 4),
                category_id=int(state["category_map"].get(it)) if pd.notna(state["category_map"].get(it)) else None,
                candidate_source=item_src_map.get(it, "item_cf"),
            )
            for r, it in enumerate(final_item_ids, 1)
        ]
    else:
        # Pure Stage 2 output
        recs = [
            RecommendationItem(
                rank=r,
                item_id=c["itemid"],
                score=round(float(c["score"]), 4),
                category_id=int(state["category_map"].get(c["itemid"])) if pd.notna(state["category_map"].get(c["itemid"])) else None,
                candidate_source=c["source"],
            )
            for r, c in enumerate(ranked_list[:k], 1)
        ]

    latency = (time.time() - t0) * 1000
    return RecommendationResponse(
        user_id=user_id,
        model_version=model_version,
        is_cold_start=is_cold_start,
        user_history_count=len(user_history),
        recommendations=recs,
        latency_ms=round(latency, 2),
    )
