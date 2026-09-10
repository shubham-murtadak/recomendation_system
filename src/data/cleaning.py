"""
Data Cleaning Module
====================
Converts timestamps, extracts temporal features, removes invalid records,
and merges item category information.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Unix millisecond timestamps to datetime.
    Extract temporal features: date, hour, day_of_week, week.
    """
    print("Converting timestamps...")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype("int8")  # 0=Monday
    df["week"] = df["datetime"].dt.isocalendar().week.astype("int8")

    time_range = df["datetime"].max() - df["datetime"].min()
    print(f"  Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Duration: {time_range.days} days")

    return df


def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove null IDs, invalid timestamps, and exact duplicates.
    Log how many records are removed at each step.
    """
    original_count = len(df)
    print(f"\nCleaning records (starting with {original_count:,})...")

    # 1. Null visitor IDs
    before = len(df)
    df = df.dropna(subset=["visitorid"])
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} rows with null visitorid")

    # 2. Null item IDs
    before = len(df)
    df = df.dropna(subset=["itemid"])
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} rows with null itemid")

    # 3. Invalid timestamps (negative or zero)
    before = len(df)
    df = df[df["timestamp"] > 0]
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} rows with invalid timestamps")

    # 4. Exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} exact duplicate rows")

    # 5. Validate event types
    valid_events = {"view", "addtocart", "transaction"}
    before = len(df)
    df = df[df["event"].isin(valid_events)]
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} rows with unknown event types")

    final_count = len(df)
    total_removed = original_count - final_count
    print(f"  Total removed: {total_removed:,} ({total_removed/original_count*100:.2f}%)")
    print(f"  Remaining: {final_count:,}")

    return df


def merge_categories(
    events: pd.DataFrame, item_categories: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-join item categories onto events.
    Items without a category will have NaN.
    """
    print(f"\nMerging item categories...")

    before_items = events["itemid"].nunique()
    events = events.merge(item_categories, on="itemid", how="left")

    items_with_cat = events["categoryid"].notna().sum()
    items_without_cat = events["categoryid"].isna().sum()

    print(f"  Events with category: {items_with_cat:,} ({items_with_cat/len(events)*100:.1f}%)")
    print(f"  Events without category: {items_without_cat:,} ({items_without_cat/len(events)*100:.1f}%)")

    return events


def run_cleaning(
    events: pd.DataFrame, item_categories: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline.
    
    Returns cleaned DataFrame with temporal features and categories.
    """
    print("=" * 60)
    print("PHASE 2.2: DATA CLEANING")
    print("=" * 60)

    events = convert_timestamps(events)
    events = remove_invalid_records(events)
    events = merge_categories(events, item_categories)

    # Save cleaned output
    processed_dir = Path(config["data"]["processed_dir"])
    output_path = processed_dir / "events_clean.parquet"
    events.to_parquet(output_path, index=False)

    print(f"\nSaved cleaned events to {output_path}")
    print(f"  Shape: {events.shape}")
    print(f"  Unique users: {events['visitorid'].nunique():,}")
    print(f"  Unique items: {events['itemid'].nunique():,}")
    print("Cleaning complete.\n")

    return events


if __name__ == "__main__":
    from ingestion import load_config, run_ingestion

    config = load_config()
    events, _, item_categories = run_ingestion(config)
    run_cleaning(events, item_categories, config)
