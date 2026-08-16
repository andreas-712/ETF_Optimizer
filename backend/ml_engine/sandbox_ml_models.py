"""
Run training, backtesting, or live inference from the terminal.
Run with:
python3 -m ml_engine.sandbox_ml_models
"""

from pathlib import Path
import asyncio
import pandas as pd

from ml_engine.live_inference import predict_tickers, run_live_inference
from ml_engine.model_orchestrator import MODELS, load_models
from ml_engine.predictor import FEATURE_COLUMNS
from ml_engine.train import train_return_predictor, train_volatility_predictor
from ml_engine.exceptions import datasetFormationError, faultyDatasetError

TRAINING_FILE_PATH = Path(__file__).resolve().parent / "batch_data" / "training_file.json"
BACKTEST_OUTPUT_PATH = Path(__file__).resolve().parent / "test_results" / "backtest_returns.txt"
BACKTEST_TEST_SIZE = 0.20
LIVE_INFERENCE_INPUTS = {
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.5,
    },
    "sizes": ["big-cap"],
    "blacklisted": [],
    "min_pool": 30,
    "max_pool": 50,
}
# Add or remove ticker: sector pairs here for direct live predictions.
DIRECT_TICKER_SECTORS = {
    "NBIS": "technology",
    "TER": "technology",
    "PLUG": "energy",
    "SOFI": "technology",
    "HOOD": "technology",
    "MU": "technology",
}

DIRECT_TICKER_HORIZON_DAYS = 20

# Set one state to "Y" when its workflow is ready to run
WORKFLOW_STATES = {
    "training": "N",
    "backtesting_inference": "Y",
    "output_file": "Y",
    "live_inference": "N",
    "direct_ticker_inference": "N",
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
        raise datasetFormationError("Backtest test_size must be between 0 and 1")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    unique_dates = sorted(result["date"].unique())
    cutoff_index = int(len(unique_dates) * (1 - test_size))
    if cutoff_index == 0 or cutoff_index >= len(unique_dates):
        raise datasetFormationError("Not enough distinct dates for a chronological train/test split")

    cutoff_date = unique_dates[cutoff_index]
    embargo_index = max(0, cutoff_index - horizon_days)
    embargo_date = unique_dates[embargo_index]
    train_df = result[result["date"] < embargo_date].copy()
    test_df = result[result["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise datasetFormationError("Chronological split produced an empty train or test set")

    return train_df, test_df


def write_backtest_output(summary: pd.DataFrame, tests: pd.DataFrame) -> None:
    """Write the backtesting report for returns only."""
    BACKTEST_OUTPUT_PATH.parent.mkdir(exist_ok = True)
    overall_accuracy = (tests["directionality"] == "correct").mean() * 100
    return_summary = summary[
        ["horizon_days", "train_rows", "test_rows", "return_directional_accuracy_percent"]
    ]

    with BACKTEST_OUTPUT_PATH.open("w", encoding = "utf-8") as output_file:
        output_file.write("Return backtest results\n\n")
        output_file.write(
            f"Overall directional accuracy: {overall_accuracy:.4f}% "
            f"({len(tests)} tests)\n\n"
        )
        output_file.write("Metrics by horizon\n")
        output_file.write(return_summary.to_string(index = False))
        output_file.write("\n\nIndividual return tests\n")
        output_file.write(tests.to_string(index = False))
        output_file.write("\n")


def run_backtesting_inference(output_file: bool = False) -> pd.DataFrame:
    """Train on earlier data and print accuracy with future data."""
    full_df = pd.read_json(TRAINING_FILE_PATH)
    results = []
    individual_tests = []

    # Run inference per timeline given unfiltered training rows (contains all timelines)
    for timeline, _ in MODELS.items():
        horizon_df = full_df[full_df["prediction_horizon_days"] == timeline].copy()
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
        directionality_correct = (
            (test_df["future_return_outcome"] >= 0) == (return_predictions >= 0)
        )
        results.append(
            {
                "horizon_days": timeline,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_start_date": test_df["date"].min().date().isoformat(),
                "test_end_date": test_df["date"].max().date().isoformat(),
                # MDA % = 1/n * sum(predicted_movement_direction == actual_movement_direction) * 100
                # Mean directional accuracy measures the rate at which the predicted price direction is correct
                "return_directional_accuracy_percent": (directionality_correct.mean() * 100),
                "volatility_rmse_percent": (((test_df["future_volatility_outcome"] - volatility_predictions).pow(2).mean()) ** 0.5 * 100),
            }
        )
        if output_file:
            individual_tests.append(
                pd.DataFrame(
                    {
                        "horizon_days": timeline,
                        "date": test_df["date"].dt.date,
                        "ticker": test_df["ticker"],
                        "predicted_return_percent": return_predictions * 100,
                        "actual_return_percent": test_df["future_return_outcome"] * 100,
                        "directionality": directionality_correct.map(
                            {True: "correct", False: "incorrect"}
                        ),
                    }
                )
            )

    summary = pd.DataFrame(results).round(
        {
            "return_directional_accuracy_percent": 4,
            "volatility_rmse_percent": 4,
        }
    )
    if output_file:
        tests = pd.concat(individual_tests, ignore_index=True).sort_values(
            ["horizon_days", "date", "ticker"]
        )
        tests = tests.round({"predicted_return_percent": 4, "actual_return_percent": 4})
        write_backtest_output(summary, tests)

    return summary



def run_direct_ticker_inference(ticker_sectors, ticker_horizon) -> dict:
    """Predict the configured ticker: sector pairs without screening a pool."""
    if not ticker_sectors:
        print("No direct ticker: sector pairs were configured.")
        return

    load_models()
    predictions = asyncio.run(
        predict_tickers(
            ticker_horizon,
            ticker_sectors,
        )
    )

    return predictions


def print_predictions(
    heading: str,
    predictions: dict[str, dict[str, float | int]],
) -> None:
    """Print live prediction results in a consistent tabular format."""
    if not predictions:
        print(f"No {heading.lower()} were returned.")
        return

    rows = [
        {"ticker": ticker, **prediction}
        for ticker, prediction in predictions.items()
    ]
    results = pd.DataFrame(rows).sort_values("ticker").round(
        {"return": 4, "volatility": 4}
    )
    print(f"\n{heading}:")
    print(results.to_string(index=False))


def main() -> None:
    """Dispatch each enabled ML workflow state."""
    if WORKFLOW_STATES["training"] == "Y":
        run_training()

    if WORKFLOW_STATES["backtesting_inference"] == "Y":
        df_summary = run_backtesting_inference(
            output_file=WORKFLOW_STATES["output_file"] == "Y"
        )
        print("\nChronological backtest results:")
        print(df_summary.to_string(index = False))

    if WORKFLOW_STATES["live_inference"] == "Y":
        preds = run_live_inference(LIVE_INFERENCE_INPUTS)
        print_predictions("Live predictions", preds)

    if WORKFLOW_STATES["direct_ticker_inference"] == "Y":
        preds = run_direct_ticker_inference(DIRECT_TICKER_SECTORS, DIRECT_TICKER_HORIZON_DAYS)
        print_predictions("Direct ticker predictions", preds)


if __name__ == "__main__":
    main()
