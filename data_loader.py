"""
CIGA - Customer Intent Gap Analyzer
====================================
Data Loader Module
Course: Big Data Computing for Business Analytics (MGT1062)
Team: Thejesh - 23MIA1033

Handles Kaggle dataset download, chunked loading (Big Data approach),
preprocessing, and caching for fast repeated runs.
"""

import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(DATA_DIR, "processed_data.pkl")
SAMPLE_ROWS = 600_000          # rows to keep for demo (≈ 600 K events)
CHUNK_SIZE  = 100_000          # read CSV in 100 K row chunks (big-data style)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Kaggle Download ───────────────────────────────────────────────────────────
def download_dataset() -> str:
    """
    Download the ecommerce behaviour dataset from Kaggle using kagglehub.
    Returns the local directory path containing the CSV files.
    """
    logger.info("Downloading dataset from Kaggle …")
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "mkechinov/ecommerce-behavior-data-from-multi-category-store"
        )
        logger.info("Dataset path: %s", path)
        return path
    except Exception as exc:
        logger.error("Kaggle download failed: %s", exc)
        raise


# ── Big-Data Chunked Loader ───────────────────────────────────────────────────
def load_raw(data_path: str, sample_size: int = SAMPLE_ROWS) -> pd.DataFrame:
    """
    Load CSV files in chunks (Big Data pattern) and return a sampled DataFrame.
    Priority: Oct-2019 file first, then Nov-2019 if more rows needed.
    """
    csv_files = sorted(Path(data_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    logger.info("Found %d CSV file(s): %s", len(csv_files),
                [f.name for f in csv_files])

    chunks = []
    rows_collected = 0

    for csv_file in csv_files:
        if rows_collected >= sample_size:
            break
        logger.info("Reading %s in %d-row chunks …", csv_file.name, CHUNK_SIZE)

        reader = pd.read_csv(csv_file, chunksize=CHUNK_SIZE,
                             low_memory=False)
        for chunk in tqdm(reader, desc=f"  {csv_file.name}", unit="chunk"):
            chunks.append(chunk)
            rows_collected += len(chunk)
            if rows_collected >= sample_size:
                break

    df = pd.concat(chunks, ignore_index=True).iloc[:sample_size]
    logger.info("Loaded %d raw rows from %d chunk(s).", len(df), len(chunks))
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the raw ecommerce DataFrame.
    Adds derived columns needed by the Intent Analyzer.
    """
    logger.info("Preprocessing %d rows …", len(df))

    # --- datetime -----------------------------------------------------------
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df.dropna(subset=["event_time"], inplace=True)
    df["date"]        = df["event_time"].dt.date
    df["hour"]        = df["event_time"].dt.hour
    df["day_of_week"] = df["event_time"].dt.day_name()

    # --- category -----------------------------------------------------------
    df["category_code"] = df["category_code"].fillna("unknown")
    df["category_main"] = df["category_code"].apply(
        lambda x: x.split(".")[0].strip() if pd.notna(x) and x != "unknown" else "unknown"
    )
    df["category_sub"] = df["category_code"].apply(
        lambda x: ".".join(x.split(".")[:2]) if "." in str(x) else x
    )

    # --- numeric -----------------------------------------------------------
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["price"] = df["price"].clip(lower=0)

    # --- price tier ---------------------------------------------------------
    df["price_tier"] = pd.cut(
        df["price"],
        bins=[0, 10, 50, 100, 500, float("inf")],
        labels=["< $10", "$10–50", "$50–100", "$100–500", "> $500"],
        right=True,
    ).astype(str)

    # --- brand -------------------------------------------------------------
    df["brand"] = df["brand"].fillna("unknown").str.lower().str.strip()

    # --- IDs ----------------------------------------------------------------
    df["user_id"]      = df["user_id"].astype(str)
    df["product_id"]   = df["product_id"].astype(str)
    df["user_session"] = df["user_session"].astype(str)

    # --- event_type validation ---------------------------------------------
    valid_events = {"view", "cart", "remove_from_cart", "purchase"}
    df = df[df["event_type"].isin(valid_events)].copy()

    logger.info(
        "Preprocessed OK  →  %d rows | events: %s",
        len(df),
        df["event_type"].value_counts().to_dict(),
    )
    return df.reset_index(drop=True)


# ── Public Entry Points ───────────────────────────────────────────────────────
def get_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Return a processed DataFrame.
    Uses a local pickle cache so repeated runs are instant.
    Pass force_reload=True to re-download and reprocess.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_reload and os.path.exists(CACHE_FILE):
        logger.info("Loading cached data from %s …", CACHE_FILE)
        with open(CACHE_FILE, "rb") as fh:
            df = pickle.load(fh)
        logger.info("Cache loaded: %d rows.", len(df))
        return df

    data_path = download_dataset()
    df = load_raw(data_path)
    df = preprocess(df)

    logger.info("Saving cache to %s …", CACHE_FILE)
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cache saved.")
    return df


# ── Structured Console Output ─────────────────────────────────────────────────
def print_structured_summary(df: pd.DataFrame) -> None:
    """Print a structured summary of the loaded dataset to stdout."""
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  CIGA DATA LOADER — STRUCTURED OUTPUT SUMMARY")
    print(f"  Course: MGT1062  |  Team: Thejesh - 23MIA1033")
    print(f"{'═'*60}")

    print(f"\n{sep}")
    print("  DATASET SHAPE")
    print(sep)
    print(f"  Total Rows      : {len(df):>12,}")
    print(f"  Total Columns   : {df.shape[1]:>12}")
    print(f"  Unique Users    : {df['user_id'].nunique():>12,}")
    print(f"  Unique Sessions : {df['user_session'].nunique():>12,}")
    print(f"  Unique Products : {df['product_id'].nunique():>12,}")
    print(f"  Date Range      : {df['date'].min()}  →  {df['date'].max()}")

    print(f"\n{sep}")
    print("  EVENT TYPE DISTRIBUTION")
    print(sep)
    for evt, cnt in df["event_type"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"  {evt:<22}: {cnt:>10,}  ({pct:5.1f}%)")

    print(f"\n{sep}")
    print("  TOP 5 CATEGORIES BY VIEWS")
    print(sep)
    top_cats = (
        df[df["event_type"] == "view"]["category_main"]
        .value_counts()
        .head(5)
    )
    for cat, cnt in top_cats.items():
        print(f"  {cat:<22}: {cnt:>10,}")

    print(f"\n{sep}")
    print("  PRICE TIER DISTRIBUTION (all events)")
    print(sep)
    for tier, cnt in df["price_tier"].value_counts().items():
        print(f"  {tier:<22}: {cnt:>10,}")

    print(f"\n{'═'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    df = get_data(force_reload=force)
    print_structured_summary(df)
