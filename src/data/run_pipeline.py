"""
Data Pipeline Runner
====================
Orchestrates the full data engineering pipeline:
    Ingestion -> Cleaning -> Sessionization -> Temporal Split
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.ingestion import load_config, run_ingestion
from src.data.cleaning import run_cleaning
from src.data.sessionization import run_sessionization


def main():
    config = load_config()

    # Phase 2.1: Ingestion
    events, category_tree, item_categories = run_ingestion(config)

    # Phase 2.2: Cleaning
    events_clean = run_cleaning(events, item_categories, config)

    # Phase 2.3: Sessionization & Temporal Split
    train, test = run_sessionization(events_clean, config)

    print("=" * 60)
    print("DATA PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Outputs in {config['data']['processed_dir']}/:")
    print(f"  train_events.parquet  ({len(train):,} rows)")
    print(f"  test_events.parquet   ({len(test):,} rows)")


if __name__ == "__main__":
    main()
