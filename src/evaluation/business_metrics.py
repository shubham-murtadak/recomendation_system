"""
Business Metrics Module
=======================
Computes business-level metrics: View Rate, Cart Rate, Purchase Rate.
"""

import pandas as pd


def compute_business_metrics(
    user_recommendations: dict,
    test_events: pd.DataFrame,
    k: int,
) -> dict:
    """
    Compute business metrics based on what users actually did
    with the recommended items in the test period.
    
    Metrics:
        - view_rate: fraction of recommended items the user viewed
        - cart_rate: fraction of recommended items the user added to cart
        - purchase_rate: fraction of recommended items the user purchased
    """
    # Build ground truth per event type
    test_views = set(
        zip(
            test_events[test_events["event"] == "view"]["visitorid"],
            test_events[test_events["event"] == "view"]["itemid"],
        )
    )
    test_carts = set(
        zip(
            test_events[test_events["event"] == "addtocart"]["visitorid"],
            test_events[test_events["event"] == "addtocart"]["itemid"],
        )
    )
    test_purchases = set(
        zip(
            test_events[test_events["event"] == "transaction"]["visitorid"],
            test_events[test_events["event"] == "transaction"]["itemid"],
        )
    )

    total_recs = 0
    total_views = 0
    total_carts = 0
    total_purchases = 0

    for user_id, rec_items in user_recommendations.items():
        top_k = rec_items[:k]
        total_recs += len(top_k)

        for item_id in top_k:
            if (user_id, item_id) in test_views:
                total_views += 1
            if (user_id, item_id) in test_carts:
                total_carts += 1
            if (user_id, item_id) in test_purchases:
                total_purchases += 1

    metrics = {
        "view_rate": total_views / total_recs if total_recs > 0 else 0.0,
        "cart_rate": total_carts / total_recs if total_recs > 0 else 0.0,
        "purchase_rate": total_purchases / total_recs if total_recs > 0 else 0.0,
        "total_recommendations": total_recs,
    }

    return metrics
