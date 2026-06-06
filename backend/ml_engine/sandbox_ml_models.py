'''
- Trains and tests models with dummy Gemini values.
- Outputs comparison graphs for 20, 90, 360 day windows
for volatility and return predictions.
- Outputs a TSV block to log parameters and accuracy
'''
from pathlib import Path
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.market_data_collection import fetch_ticker_data
from ml_engine.train import (
    build_training_frame,
    save_model,
    train_return_predictor,
    train_risk_predictor,
)


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "XOM", "UNH", "TSLA"]
PREDICTION_HORIZON_DAYS = [20, 90, 360]
PLOT_DIR = Path(__file__).resolve().parent / "plots"
LOOKBACK_YEARS = 6
TEST_SIZE = 0.2
ROLLING_VOLATILITY_WINDOW = 10
KALMAN_Q = 1e-5
KALMAN_R = 1e-2
RATINGS_USED = "n"
RISK_USED = "n"
RETURN_TARGET_TYPE = "pct_change_shifted"
VOLATILITY_TARGET_TYPE = "rolling_std_shifted"
FEATURE_COLUMNS = [
    "price_trend_deviation",
    "rolling_volatility",
    "gemini_sentiment_score",
    "gemini_risk_flag",
]

# Dummy sandbox values.
# sentiment: -1.0 bearish, 0.0 neutral, 1.0 bullish
# risk_flag: 0 low, 1 medium, 2 high
DUMMY_RATINGS_1 = {
    "AAPL": {"sentiment": 0.40, "risk": 0},
    "MSFT": {"sentiment": 0.80, "risk": 0},
    "GOOGL": {"sentiment": 0.70, "risk": 0},
    "AMZN": {"sentiment": 0.70, "risk": 0},
    "NVDA": {"sentiment": 0.30, "risk": 1},
    "META": {"sentiment": 0.50, "risk": 1},
    "JPM": {"sentiment": 0.60, "risk": 0},
    "XOM": {"sentiment": -0.10, "risk": 1},
    "UNH": {"sentiment": 0.70, "risk": 0},
    "TSLA": {"sentiment": -0.20, "risk": 2},
}

DUMMY_RATINGS_2 = {
    "AAPL": {"sentiment": 0.00, "risk": 1},
    "MSFT": {"sentiment": 0.00, "risk": 1},
    "GOOGL": {"sentiment": 0.00, "risk": 1},
    "AMZN": {"sentiment": 0.00, "risk": 1},
    "NVDA": {"sentiment": 0.00, "risk": 1},
    "META": {"sentiment": 0.00, "risk": 1},
    "JPM": {"sentiment": 0.00, "risk": 1},
    "XOM": {"sentiment": 0.00, "risk": 1},
    "UNH": {"sentiment": 0.00, "risk": 1},
    "TSLA": {"sentiment": 0.00, "risk": 1},
}


def chronological_train_test_split(
    df: pd.DataFrame,
    horizon_days: int,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df["date"].unique())
    cutoff_index = int(len(unique_dates) * (1 - test_size))
    cutoff_date = unique_dates[cutoff_index]
    embargo_index = max(0, cutoff_index - horizon_days)
    embargo_start_date = unique_dates[embargo_index]

    train_df = df[df["date"] < embargo_start_date]
    test_df = df[df["date"] >= cutoff_date]

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Date split produced an empty train or test set. "
            "Use more lookback data or a shorter prediction horizon."
        )

    return train_df, test_df


def latest_prediction_report(
    df: pd.DataFrame, return_model, risk_model, horizon_days: int
) -> pd.DataFrame:
    latest_rows = df.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()
    X_latest = latest_rows[FEATURE_COLUMNS]

    latest_rows["prediction_horizon_days"] = horizon_days
    latest_rows["predicted_return"] = return_model.predict(X_latest)
    latest_rows["predicted_volatility"] = risk_model.predict(X_latest)

    return latest_rows[
        [
            "date",
            "ticker",
            "prediction_horizon_days",
            "future_return_outcome",
            "predicted_return",
            "future_volatility_outcome",
            "predicted_volatility",
            "gemini_sentiment_score",
            "gemini_risk_flag",
        ]
    ].sort_values("ticker")


def return_test_mae_percent(test_df: pd.DataFrame, return_model) -> float:
    predictions = return_model.predict(test_df[FEATURE_COLUMNS])
    mae = np.mean(np.abs(test_df["future_return_outcome"] - predictions))
    return mae * 100


def volatility_test_mae_percent(test_df: pd.DataFrame, risk_model) -> float:
    predictions = risk_model.predict(test_df[FEATURE_COLUMNS])
    mae = np.mean(np.abs(test_df["future_volatility_outcome"] - predictions))
    return mae * 100


def plot_return_predictions(test_df: pd.DataFrame, return_model, horizon_days: int) -> Path:
    test_df = test_df.copy()
    test_df["predicted_return"] = return_model.predict(test_df[FEATURE_COLUMNS])

    tickers = sorted(test_df["ticker"].unique())
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=False)
    axes = axes.flatten()

    for ax, ticker in zip(axes, tickers):
        ticker_df = test_df[test_df["ticker"] == ticker].sort_values("date")
        ax.plot(ticker_df["date"], ticker_df["future_return_outcome"] * 100, label="Actual")
        ax.plot(ticker_df["date"], ticker_df["predicted_return"] * 100, label="Predicted")
        ax.set_title(ticker)
        ax.set_ylabel("Return %")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(tickers):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{horizon_days}-Trading-Day Return: Predicted vs Actual", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"return_predictions_{horizon_days}d.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_volatility_predictions(test_df: pd.DataFrame, risk_model, horizon_days: int) -> Path:
    test_df = test_df.copy()
    test_df["predicted_volatility"] = risk_model.predict(test_df[FEATURE_COLUMNS])

    tickers = sorted(test_df["ticker"].unique())
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=False)
    axes = axes.flatten()

    for ax, ticker in zip(axes, tickers):
        ticker_df = test_df[test_df["ticker"] == ticker].sort_values("date")
        ax.plot(ticker_df["date"], ticker_df["future_volatility_outcome"] * 100, label="Actual")
        ax.plot(ticker_df["date"], ticker_df["predicted_volatility"] * 100, label="Predicted")
        ax.set_title(ticker)
        ax.set_ylabel("Volatility %")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(tickers):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{horizon_days}-Trading-Day Volatility: Predicted vs Actual", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"volatility_predictions_{horizon_days}d.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def train_models_for_horizon(market_df: pd.DataFrame, horizon_days: int) -> dict:
    print(f"\nTraining {horizon_days}-trading-day models")
    training_df = build_training_frame(
        market_df,
        horizon_days=horizon_days,
        gemini_ratings=DUMMY_RATINGS_2,
        feature_columns=FEATURE_COLUMNS,
        rolling_volatility_window=ROLLING_VOLATILITY_WINDOW,
        kalman_q=KALMAN_Q,
        kalman_r=KALMAN_R,
    )
    train_df, test_df = chronological_train_test_split(training_df, horizon_days)

    return_model = train_return_predictor(
        train_df,
        FEATURE_COLUMNS,
        "future_return_outcome",
    )
    risk_model = train_risk_predictor(
        train_df,
        FEATURE_COLUMNS,
        "future_volatility_outcome",
    )

    return_path = save_model(return_model, f"gbr_return_model_{horizon_days}d.pkl")
    risk_path = save_model(risk_model, f"rfr_volatility_model_{horizon_days}d.pkl")

    # Prove the saved files can be loaded again.
    loaded_return_model = joblib.load(return_path)
    loaded_risk_model = joblib.load(risk_path)

    report = latest_prediction_report(
        training_df,
        loaded_return_model,
        loaded_risk_model,
        horizon_days=horizon_days,
    )
    print("\nSaved models:")
    print(f"- {return_path}")
    print(f"- {risk_path}")
    print("\nLatest per-ticker dummy sandbox comparison:")
    print(report.to_string(index=False))

    return_plot_path = plot_return_predictions(test_df, loaded_return_model, horizon_days)
    volatility_plot_path = plot_volatility_predictions(test_df, loaded_risk_model, horizon_days)
    print(f"\nSaved return plot: {return_plot_path}")
    print(f"Saved volatility plot: {volatility_plot_path}")

    return_mae = return_test_mae_percent(test_df, loaded_return_model)
    volatility_mae = volatility_test_mae_percent(test_df, loaded_risk_model)

    return {
        "horizon_days": horizon_days,
        "training_rows": len(training_df),
        "return_test_mae_percent": return_mae,
        "volatility_test_mae_percent": volatility_mae,
        "return_model_path": return_path,
        "risk_model_path": risk_path,
        "return_plot_path": return_plot_path,
        "volatility_plot_path": volatility_plot_path,
        "gbr_n_estimator": loaded_return_model.get_params()["n_estimators"],
        "gbr_learning": loaded_return_model.get_params()["learning_rate"],
        "gbr_depth": loaded_return_model.get_params()["max_depth"],
        "gbr_subsample": loaded_return_model.get_params()["subsample"],
        "rfr_n_estimator": loaded_risk_model.get_params()["n_estimators"],
        "rfr_depth": loaded_risk_model.get_params()["max_depth"],
        "rfr_samples": loaded_risk_model.get_params()["min_samples_split"],
        "ratings?": RATINGS_USED,
        "risk?": RISK_USED,
        "kalman_Q": KALMAN_Q,
        "kalman_R": KALMAN_R,
        "lookback_years": LOOKBACK_YEARS,
        "rolling_volatility_window": ROLLING_VOLATILITY_WINDOW,
        "test_size": TEST_SIZE,
        "embargo_days": horizon_days,
        "ticker_count": len(TICKERS),
        "feature_set": "|".join(FEATURE_COLUMNS),
        "return_target_type": RETURN_TARGET_TYPE,
        "volatility_target_type": VOLATILITY_TARGET_TYPE,
        "pred_timeline_days": horizon_days,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "return_mae": return_mae,
        "volatilility_mae": volatility_mae,
    }


def print_sheet_rows(results: list[dict]) -> None:
    columns = [
        "gbr_n_estimator",
        "gbr_learning",
        "gbr_depth",
        "gbr_subsample",
        "rfr_n_estimator",
        "rfr_depth",
        "rfr_samples",
        "ratings?",
        "risk?",
        "kalman_Q",
        "kalman_R",
        "lookback_years",
        "rolling_volatility_window",
        "test_size",
        "embargo_days",
        "ticker_count",
        "feature_set",
        "return_target_type",
        "volatility_target_type",
        "pred_timeline_days",
        "train_rows",
        "test_rows",
        "return_mae",
        "volatilility_mae",
    ]
    rows = pd.DataFrame(results)[columns].round(
        {
            "gbr_learning": 4,
            "gbr_subsample": 4,
            "kalman_Q": 8,
            "kalman_R": 8,
            "test_size": 4,
            "return_mae": 2,
            "volatilility_mae": 2,
        }
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / "sandbox_model_results.tsv"
    rows.to_csv(output_path, sep="\t", index=False)

    print("\nGoogle Sheets TSV:")
    print(rows.to_csv(sep="\t", index=False).strip())
    print(f"\nSaved TSV: {output_path}")


def main():
    market_df = fetch_ticker_data(TICKERS, lookback_years = LOOKBACK_YEARS)
    if market_df.empty:
        raise RuntimeError("No market data was downloaded. Check yfinance/network access.")

    results = [
        train_models_for_horizon(market_df, horizon_days)
        for horizon_days in PREDICTION_HORIZON_DAYS
    ]

    summary = pd.DataFrame(results)
    summary["return_test_mae_percent"] = summary["return_test_mae_percent"].round(2)
    summary["volatility_test_mae_percent"] = summary["volatility_test_mae_percent"].round(2)

    print("\nMean absolute error summary on chronological test split:")
    print(
        summary[
            [
                "horizon_days",
                "training_rows",
                "return_test_mae_percent",
                "volatility_test_mae_percent",
            ]
        ].to_string(index = False)
    )
    print_sheet_rows(results)


if __name__ == "__main__":
    main()
