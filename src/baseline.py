from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/hour.csv")
CUTOFF = "2012-10-01"

def loadAndSort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])

    return df.sort_values(["dteday", "hr"]).reset_index(drop=True)

def dateSpan(dates: pd.Series) -> str:
    return f"{dates.min().date()} to {dates.max().date()}"

def printMetrics(metrics: dict) -> None:
    print(f"MAE {metrics['mae']:.2f}")
    print(f"RMSE {metrics['rmse']:.2f}")
    print(f"R2 {metrics['r2']:.3f}")

def audit(df: pd.DataFrame) -> None:
    print("Data Audit")
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print(f"\nMissing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate (dteday, hr) pairs: {df.duplicated(['dteday', 'hr']).sum()}")
    print("\nTarget distribution:")
    print(df["cnt"].describe().round(2).to_string())
    print(f"cnt min/max: {df['cnt'].min()} / {df['cnt'].max()}")

    q1, q3 = df["cnt"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df["cnt"] < q1 - 1.5 * iqr) | (df["cnt"] > q3 + 1.5 * iqr)).sum()

    print(f"cnt values outside 1.5*IQR: {outliers}")
    print(f"weathersit value counts:\n{df['weathersit'].value_counts().sort_index()}")
    print(f"Date range: {dateSpan(df['dteday'])}")


def timeSplit(df: pd.DataFrame, cutoff: str):
    train = df[df["dteday"] < cutoff].copy()
    test = df[df["dteday"] >= cutoff].copy()
    
    print("\n=== SPLIT ===")
    print(f"Cutoff: {cutoff}")
    print(f"Train rows: {len(train)} ({dateSpan(train['dteday'])})")
    print(f"Test rows:  {len(test)} ({dateSpan(test['dteday'])})")
    return train, test


def hourMeanBaseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    hour_means = train.groupby("hr")["cnt"].mean()
    global_mean = train["cnt"].mean()

    y_pred = test["hr"].map(hour_means).fillna(global_mean)
    y_true = test["cnt"]

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def globalMeanBaseline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    pred = np.full(len(test), train["cnt"].mean())
    y_true = test["cnt"]

    return {
        "mae": mean_absolute_error(y_true, pred),
        "rmse": np.sqrt(mean_squared_error(y_true, pred)),
        "r2": r2_score(y_true, pred),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}. Download hour.csv into data/ first.")

    df = loadAndSort(DATA_PATH)
    audit(df)

    drop_cols = ["instant", "casual", "registered"]
    print(f"\nColumns dropped (index or leakage): {drop_cols}")
    print("dteday used only for ordering and the time split.")

    train, test = timeSplit(df, CUTOFF)

    hour_metrics = hourMeanBaseline(train, test)
    global_metrics = globalMeanBaseline(train, test)

    print("\n=== BASELINE RESULTS (test set) ===")
    print("Hour-of-day mean (primary baseline):")
    printMetrics(hour_metrics)
    print("Global training mean (reference):")
    printMetrics(global_metrics)


if __name__ == "__main__":
    main()
