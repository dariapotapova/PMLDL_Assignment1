"""
Stage 1: Data Engineering — Palmer Penguins.

Loads raw data, cleans it (drops unrecoverable rows, imputes missing
categorical values), removes outliers, and splits into train/test sets.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "penguins.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

# Raw data is cached locally: the pipeline runs every 5 minutes,
# so we avoid hitting the network on every cycle.
RAW_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
RAW_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

TARGET = "species"
NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CAT_COLS = ["island", "sex"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    if RAW_PATH.exists():
        log.info("Using cached raw data: %s", RAW_PATH)
        df = pd.read_csv(RAW_PATH)
    else:
        log.info("Downloading penguins from %s", RAW_URL)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(RAW_URL)
        df.to_csv(RAW_PATH, index=False)
    log.info("Raw data: %d rows, %d cols", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[[TARGET, *CAT_COLS, *NUMERIC_COLS]].copy()

    # Rows missing the target or measurements cannot be recovered -> drop.
    # Missing categorical 'sex' CAN be reasonably restored -> imputed below.
    df = df.dropna(subset=[TARGET, *NUMERIC_COLS])
    n_dropped = n_before - len(df)
    n_missing_sex = int(df["sex"].isna().sum())

    # Impute with the mode WITHIN each species: sex ratio differs across
    # species, so a global mode would distort the distribution.
    df["sex"] = df.groupby(TARGET)["sex"].transform(lambda s: s.fillna(s.mode()[0]))

    log.info("Dropped %d rows with missing target/features", n_dropped)
    log.info("Imputed %d missing 'sex' with per-species mode", n_missing_sex)
    log.info("After cleaning: %d rows", len(df))
    return df


def remove_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    # Tukey's rule: a value outside [Q1 - k*IQR, Q3 + k*IQR] is an outlier.
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask &= df[col].between(lower, upper)
        log.info("%s: keep [%.1f, %.1f]", col, lower, upper)
    df = df[mask]
    log.info("Outliers removed: %d -> %d rows", n_before, len(df))
    return df


def split_and_save(df: pd.DataFrame) -> None:
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        # Classes are imbalanced — stratify keeps
        # class proportions identical in both splits.
        stratify=df[TARGET]
    )
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    log.info("Saved train (%d rows) -> %s", len(train), TRAIN_PATH)
    log.info("Saved test  (%d rows) -> %s", len(test), TEST_PATH)


def main() -> None:
    df = load_raw_data()
    df = clean_data(df)
    df = remove_outliers(df)
    split_and_save(df)
    log.info("Stage 1 done.")


if __name__ == "__main__":
    main()