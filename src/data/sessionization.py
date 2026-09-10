"""
Sessionization Module
=====================
Creates browsing sessions from raw events using a configurable inactivity gap.
Assigns interaction weights. Performs temporal train/test split.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_sessions(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create browsing sessions based on inactivity gap.
    
    Logic:
        1. Sort events by (visitorid, datetime)
        2. For each user, if the time gap between consecutive events
           exceeds the configured threshold, start a new session.
        3. Assign a unique session_id to each session.
    """
    gap_minutes = config["sessionization"]["gap_minutes"]
    gap_threshold = pd.Timedelta(minutes=gap_minutes)

    print(f"Creating sessions (gap threshold: {gap_minutes} min)...")

    # Sort by user and time
    events = events.sort_values(["visitorid", "datetime"]).reset_index(drop=True)

    # Calculate time difference between consecutive events per user
    events["time_diff"] = events.groupby("visitorid")["datetime"].diff()

    # Mark session boundaries: first event of a user OR gap > threshold
    events["new_session"] = (events["time_diff"].isna()) | (
        events["time_diff"] > gap_threshold
    )

    # Assign session IDs using cumulative sum of session boundaries
    events["session_id"] = events["new_session"].cumsum()

    # Clean up temporary columns
    events = events.drop(columns=["time_diff", "new_session"])

    n_sessions = events["session_id"].nunique()
    n_users = events["visitorid"].nunique()

    print(f"  Created {n_sessions:,} sessions from {n_users:,} users")
    print(f"  Avg sessions/user: {n_sessions / n_users:.2f}")

    # Session length stats
    session_lengths = events.groupby("session_id").size()
    print(f"  Session length stats:")
    print(f"    Mean:   {session_lengths.mean():.1f}")
    print(f"    Median: {session_lengths.median():.1f}")
    print(f"    Max:    {session_lengths.max()}")

    return events


def add_interaction_scores(events: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add interaction_score column based on configurable event weights.
    
    Default weights:
        view=1, addtocart=3, transaction=5
    """
    weights = config["interaction_weights"]

    print(f"\nAdding interaction scores: {weights}")

    events["interaction_score"] = events["event"].map(weights).astype("int8")

    # Sanity check: no unmapped events
    unmapped = events["interaction_score"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} events could not be mapped to weights!")
        events["interaction_score"] = events["interaction_score"].fillna(1)

    print(f"  Score distribution:")
    print(events.groupby("event")["interaction_score"].first().to_string())

    return events


def temporal_split(events: pd.DataFrame, config: dict) -> tuple:
    """
    Split data temporally: first X% by time = train, rest = test.
    
    This prevents temporal leakage — we never train on future data.
    """
    train_ratio = config["temporal_split"]["train_ratio"]

    min_time = events["datetime"].min()
    max_time = events["datetime"].max()
    total_duration = max_time - min_time

    split_time = min_time + (total_duration * train_ratio)

    print(f"\nTemporal split at {split_time} (train_ratio={train_ratio})")

    train = events[events["datetime"] < split_time].copy()
    test = events[events["datetime"] >= split_time].copy()

    print(f"  Train: {len(train):,} events ({len(train)/len(events)*100:.1f}%)")
    print(f"    Time: {train['datetime'].min()} to {train['datetime'].max()}")
    print(f"    Users: {train['visitorid'].nunique():,}")
    print(f"    Items: {train['itemid'].nunique():,}")

    print(f"  Test:  {len(test):,} events ({len(test)/len(events)*100:.1f}%)")
    print(f"    Time: {test['datetime'].min()} to {test['datetime'].max()}")
    print(f"    Users: {test['visitorid'].nunique():,}")
    print(f"    Items: {test['itemid'].nunique():,}")

    # Check for user overlap (expected and healthy)
    train_users = set(train["visitorid"].unique())
    test_users = set(test["visitorid"].unique())
    overlap = train_users & test_users
    cold_start_users = test_users - train_users

    print(f"  User overlap (train & test): {len(overlap):,}")
    print(f"  Cold-start users (test only): {len(cold_start_users):,}")

    return train, test


def run_sessionization(events: pd.DataFrame, config: dict) -> tuple:
    """
    Run the full sessionization pipeline.
    
    Returns:
        (train_events, test_events)
    """
    print("=" * 60)
    print("PHASE 2.3: SESSIONIZATION & TEMPORAL SPLIT")
    print("=" * 60)

    events = create_sessions(events, config)
    events = add_interaction_scores(events, config)
    train, test = temporal_split(events, config)

    # Save outputs
    processed_dir = Path(config["data"]["processed_dir"])

    train.to_parquet(processed_dir / "train_events.parquet", index=False)
    test.to_parquet(processed_dir / "test_events.parquet", index=False)

    print(f"\nSaved train/test splits to {processed_dir}/")
    print("Sessionization complete.\n")

    return train, test


if __name__ == "__main__":
    from ingestion import load_config
    
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    
    print("Loading cleaned events...")
    events = pd.read_parquet(processed_dir / "events_clean.parquet")
    
    run_sessionization(events, config)
