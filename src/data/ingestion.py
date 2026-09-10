"""
Data Ingestion Module
=====================
Loads raw RetailRocket CSV files with optimized dtypes.
Extracts item-to-category mappings from item_properties files.
Validates schema and logs basic stats.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_events(config: dict) -> pd.DataFrame:
    """
    Load events.csv with memory-optimized dtypes.
    
    Returns DataFrame with columns:
        timestamp, visitorid, event, itemid, transactionid
    """
    raw_dir = Path(config["data"]["raw_dir"])
    filepath = raw_dir / config["data"]["events_file"]

    print(f"Loading events from {filepath}...")

    dtypes = {
        "visitorid": "int32",
        "event": "category",
        "itemid": "int32",
        "transactionid": "float64",  # Has NaN values for non-transaction events
    }

    df = pd.read_csv(filepath, dtype=dtypes)

    # Validate
    assert "timestamp" in df.columns, "Missing 'timestamp' column"
    assert "visitorid" in df.columns, "Missing 'visitorid' column"
    assert "event" in df.columns, "Missing 'event' column"
    assert "itemid" in df.columns, "Missing 'itemid' column"

    print(f"  Loaded {len(df):,} events")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Events distribution:")
    print(df["event"].value_counts().to_string(header=False))

    return df


def load_category_tree(config: dict) -> pd.DataFrame:
    """
    Load category_tree.csv containing the category hierarchy.
    
    Returns DataFrame with columns:
        categoryid, parentid
    """
    raw_dir = Path(config["data"]["raw_dir"])
    filepath = raw_dir / config["data"]["category_tree_file"]

    print(f"\nLoading category tree from {filepath}...")

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} category entries")

    return df


def extract_item_categories(config: dict) -> pd.DataFrame:
    """
    Extract item-to-category mapping from item_properties files.
    
    The item_properties files are large (~900MB combined) and contain
    many different properties. We only need rows where property == 'categoryid'.
    We read in chunks to avoid OOM.
    
    Returns DataFrame with columns:
        itemid, categoryid
    """
    raw_dir = Path(config["data"]["raw_dir"])
    filenames = config["data"]["item_properties_files"]

    print("\nExtracting item categories from item_properties files...")

    category_rows = []

    for filename in filenames:
        filepath = raw_dir / filename
        print(f"  Processing {filepath}...")

        # Read in chunks to handle the large file size
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=500_000,
            dtype={"itemid": "int32", "property": "str", "value": "str"},
        )

        for chunk in chunk_iter:
            # Only keep categoryid rows
            cat_chunk = chunk[chunk["property"] == "categoryid"][
                ["timestamp", "itemid", "value"]
            ].copy()
            if len(cat_chunk) > 0:
                category_rows.append(cat_chunk)

    if not category_rows:
        print("  WARNING: No categoryid properties found!")
        return pd.DataFrame(columns=["itemid", "categoryid"])

    df = pd.concat(category_rows, ignore_index=True)

    # Items can have time-varying categories. Take the LATEST category assignment.
    df = df.sort_values("timestamp").drop_duplicates(subset="itemid", keep="last")
    df = df.rename(columns={"value": "categoryid"})
    df["categoryid"] = pd.to_numeric(df["categoryid"], errors="coerce").astype(
        "Int32"
    )
    df = df[["itemid", "categoryid"]].dropna(subset=["categoryid"])

    print(f"  Extracted categories for {len(df):,} unique items")

    return df


def run_ingestion(config: dict) -> tuple:
    """
    Run the full ingestion pipeline.
    
    Returns:
        (events_df, category_tree_df, item_categories_df)
    """
    print("=" * 60)
    print("PHASE 2.1: DATA INGESTION")
    print("=" * 60)

    events = load_events(config)
    category_tree = load_category_tree(config)
    item_categories = extract_item_categories(config)

    # Save intermediate outputs
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    events.to_parquet(processed_dir / "events_raw.parquet", index=False)
    category_tree.to_parquet(processed_dir / "category_tree.parquet", index=False)
    item_categories.to_parquet(processed_dir / "item_categories.parquet", index=False)

    print(f"\nSaved intermediate files to {processed_dir}/")
    print("Ingestion complete.\n")

    return events, category_tree, item_categories


if __name__ == "__main__":
    config = load_config()
    run_ingestion(config)
