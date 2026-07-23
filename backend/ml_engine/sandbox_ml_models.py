"""Run training, backtesting, or live inference from the terminal."""

from pathlib import Path
import asyncio

import pandas as pd

from ml_engine.live_inference import get_ticker_pool, predict_tickers
from ml_engine.model_orchestrator import MODELS, load_models
from ml_engine.predictor import FEATURE_COLUMNS
from ml_engine.train import train_return_predictor, train_volatility_predictor

TRAINING_FILE_PATH = Path(__file__).resolve().parent / "batch_data" / "training_file.json"
BACKTEST_TEST_SIZE = 0.20
LIVE_INFERENCE_INPUTS = {
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["big-cap"],
    "blacklisted": [],
    "min_pool": 10,
    "max_pool": 10,
}

# Set one state to "Y" when its workflow is ready to run
WORKFLOW_STATES = {
    "training": "N",
    "backtesting_inference": "N",
    "live_inference": "N",
}


def run_training() -> None:
    """Train the 3, 20, and 90-day models from flattened training data."""
    df = pd.read_json(TRAINING_FILE_PATH)

    for timeline, model in MODELS.items():
        model.train(df[df["prediction_horizon_days"] == timeline])


def chronological_train_test_split(
    df: pd.DataFrame,
    horizon_days: int,
    test_size: float = BACKTEST_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split final training rows for one horizon into earlier and later dates.
    Use rows from training_file.json after filtering to one horizon.
    The df should include the model features and both future outcome columns.
    """
    if not 0 < test_size < 1:
        raise ValueError("Backtest test_size must be between 0 and 1")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    unique_dates = sorted(result["date"].unique())
    cutoff_index = int(len(unique_dates) * (1 - test_size))
    if cutoff_index == 0 or cutoff_index >= len(unique_dates):
        raise ValueError("Not enough distinct dates for a chronological train/test split")

    cutoff_date = unique_dates[cutoff_index]
    embargo_index = max(0, cutoff_index - horizon_days)
    embargo_date = unique_dates[embargo_index]
    train_df = result[result["date"] < embargo_date].copy()
    test_df = result[result["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty train or test set")

    return train_df, test_df


def run_backtesting_inference() -> None:
    """Train on earlier data and print accuracy with future data."""
    full_df = pd.read_json(TRAINING_FILE_PATH)
    results = []

    for timeline, _ in MODELS.items():
        horizon_df = full_df[
            full_df["prediction_horizon_days"] == timeline
        ].copy()
        train_df, test_df = chronological_train_test_split(horizon_df, timeline)

        return_model = train_return_predictor(
            train_df,
            FEATURE_COLUMNS,
            "future_return_outcome",
            timeline,
        )
        volatility_model = train_volatility_predictor(
            train_df,
            FEATURE_COLUMNS,
            "future_volatility_outcome",
            timeline,
        )

        return_predictions = return_model.predict(test_df[FEATURE_COLUMNS])
        volatility_predictions = volatility_model.predict(test_df[FEATURE_COLUMNS])
        results.append(
            {
                "horizon_days": timeline,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_start_date": test_df["date"].min().date().isoformat(),
                "test_end_date": test_df["date"].max().date().isoformat(),
                "return_mae_percent": (
                    (test_df["future_return_outcome"] - return_predictions)
                    .abs()
                    .mean()
                    * 100
                ),
                "volatility_mae_percent": (
                    (test_df["future_volatility_outcome"] - volatility_predictions)
                    .abs()
                    .mean()
                    * 100
                ),
            }
        )

    summary = pd.DataFrame(results).round(
        {"return_mae_percent": 4, "volatility_mae_percent": 4}
    )
    print("\nChronological backtest results:")
    print(summary.to_string(index=False))


def run_live_inference() -> None:
    """Select live candidates, predict their returns and volatility, and print them."""
    load_models()
    candidates = asyncio.run(get_ticker_pool(LIVE_INFERENCE_INPUTS))
    ticker_industries = {
        candidate["ticker"]: candidate["industry"]
        for candidate in candidates
    }
    predictions = asyncio.run(
        predict_tickers(
            LIVE_INFERENCE_INPUTS["horizon_days"],
            ticker_industries,
        )
    )

    if not predictions:
        print("No live predictions were returned.")
        return

    rows = [
        {"ticker": ticker, **prediction}
        for ticker, prediction in predictions.items()
    ]
    results = pd.DataFrame(rows).sort_values("ticker").round(
        {"return": 4, "volatility": 4}
    )
    print("\nLive predictions:")
    print(results.to_string(index=False))


def main() -> None:
    """Dispatch each enabled ML workflow state."""
    if WORKFLOW_STATES["training"] == "Y":
        run_training()

    if WORKFLOW_STATES["backtesting_inference"] == "Y":
        run_backtesting_inference()

    if WORKFLOW_STATES["live_inference"] == "Y":
        run_live_inference()


if __name__ == "__main__":
    main()
