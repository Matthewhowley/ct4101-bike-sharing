from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path("data/hour.csv")
CUTOFF = "2012-10-01"
FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",]
TARGET = "cnt"
OUT_DIR = Path("reports")


def load_and_sort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df.sort_values(["dteday", "hr"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, cutoff: str):
    train = df[df["dteday"] < cutoff].copy()
    test = df[df["dteday"] >= cutoff].copy()
    return train, test


def metrics(y_true, y_pred) -> dict:
    return {
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Root Mean Square Error": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def hour_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    hour_means = train.groupby("hr")[TARGET].mean()
    global_mean = train[TARGET].mean()
    return test["hr"].map(hour_means).fillna(global_mean).to_numpy()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_and_sort(DATA_PATH)
    train, test = time_split(df, CUTOFF)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    baseline_pred = hour_mean_baseline(train, test)

    # Decision Tree
    tree = DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)

    # Random Forest
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)

    rows = [
        {"Model": "Hour of day mean (baseline)", **metrics(y_test, baseline_pred)},
        {"Model": "Decision Tree (max_depth = 10)", **metrics(y_test, tree_pred)},
        {"Model": "Random Forest (n=100, max_depth = 12)", **metrics(y_test, forest_pred)},
    ]
    results = pd.DataFrame(rows)
    results = results.round({"Mean Absolute Error": 2, "Root Mean Square Error": 2, "R2": 3})
    print(results.to_string(index=False))
    results.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    residuals = y_test.to_numpy() - forest_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(forest_pred, residuals, alpha=0.3, s=10)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted cnt")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Random Forest residuals on test set")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rf_residuals.png", dpi=120)
    plt.close()

    # Feature importance
    importances = pd.Series(forest.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rf_feature_importance.png", dpi=120)
    plt.close()

    # Residual analysis by hour and weather
    test_out = test.copy()
    test_out["pred"] = forest_pred
    test_out["residual"] = residuals
    test_out["abs_error"] = np.abs(residuals)
    by_hour = test_out.groupby("hr")["abs_error"].mean().round(1)
    by_weather = test_out.groupby("weathersit")["abs_error"].mean().round(1)
    print("\nMean absolute error by hour (RF):")
    print(by_hour.to_string())
    print("\nMean absolute error by weathersit (RF):")
    print(by_weather.to_string())
    by_hour.to_csv(OUT_DIR / "rf_mae_by_hour.csv")
    by_weather.to_csv(OUT_DIR / "rf_mae_by_weather.csv")


if __name__ == "__main__":
    main()