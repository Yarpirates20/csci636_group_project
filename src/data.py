import numpy as np
import pandas as pd
from .config import DATA, LEVEL_COLS, LOG_COLS

#Reads the processed CSVs and returns X, y for train/validation/test splits
# Also returns next-day returns for backtesting.
def load_split(kind: str, split: str) -> pd.DataFrame:
    #kind: 'baseline' | 'enhanced'; split: 'train' | 'validation' | 'test'.
    # Returns a DataFrame with date index, features, and target.
    return pd.read_csv(DATA / "processed" / kind / f"{split}.csv",
                       parse_dates=["date"], index_col="date")

# Prepares the feature matrix X from a DataFrame
def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["target"]).copy() 
    #Drop raw price levels because they are non-stationary; keep ratios and log-transformed features
    return X.drop(columns=[c for c in LEVEL_COLS if c in X]) 

# Loads features (X) and target (y) for a given split
def load_xy(kind: str, split: str):
    df = load_split(kind, split) # Returns a DataFrame with date index, features, and target.
    return make_X(df), df["target"].astype(int) 

#Loads next-day return prices from the full cleaned price file
def load_next_return() -> pd.Series:
    #Returns a Series of next-day returns (shifted -1) for backtesting.
    close = pd.read_csv(DATA / "cleaned" / "price_cleaned.csv",
                        parse_dates=["date"], index_col="date")["close"]
    return close.shift(-1) / close - 1