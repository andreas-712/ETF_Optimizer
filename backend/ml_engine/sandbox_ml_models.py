'''
- Trains and tests models with Gemini values.
- Outputs latest single predictions for 20, 90, 360 day windows.
- Outputs a TSV block to log parameters and accuracy
'''
import asyncio
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.market_data_collection import fetch_gemini_ticker_data
from ml_engine.gemini import fetch_ticker_data as fetch_gemini_ticker_data
from ml_engine.train import (
    build_model_feature_frame,
    build_training_frame,
    return_half_life_days,
    save_model,
    train_return_predictor,
    train_volatility_predictor,
)


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "XOM", "UNH", "TSLA"]
PREDICTION_HORIZON_DAYS = [20, 90, 360]
PLOT_DIR = Path(__file__).resolve().parent / "plots"
LOOKBACK_YEARS = 6
TEST_SIZE = 0.2
ROLLING_VOLATILITY_WINDOW = 10
KALMAN_Q = 1e-5
KALMAN_R = 1e-2
GEMINI_SCORING_USED = "relevance*polarity*urgency"
RETURN_TARGET_TYPE = "pct_change_shifted"
VOLATILITY_TARGET_TYPE = "rolling_std_shifted"
FEATURE_COLUMNS = [
    "price_trend_deviation",
    "rolling_volatility",
    "gemini_sentiment_score",
]

async def fetch_gemini_records(
    tickers: list[str],
    date: str,
    horizon_days: int,
) -> list[dict]:
    return await asyncio.gather(
        *[
            fetch_gemini_ticker_data(ticker, date, horizon_days)
            for ticker in tickers
        ]
    )


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
    df: pd.DataFrame,
    return_model,
    risk_model,
    horizon_days: int,
) -> pd.DataFrame:
    latest_rows = df.dropna(subset=FEATURE_COLUMNS)
    latest_rows = latest_rows.sort_values("date").groupby("ticker", as_index=False).tail(1).copy()
    X_latest = latest_rows[FEATURE_COLUMNS]

    latest_rows["prediction_horizon_days"] = horizon_days
    latest_rows["predicted_return"] = return_model.predict(X_latest)
    latest_rows["predicted_volatility"] = risk_model.predict(X_latest)

    return latest_rows[
        [
            "date",
            "ticker",
            "prediction_horizon_days",
            "predicted_return",
            "predicted_volatility",
            "gemini_sentiment_score",
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


def train_models_for_horizon(
    market_df: pd.DataFrame,
    gemini_data: list[dict],
    horizon_days: int,
) -> tuple[dict, pd.DataFrame]:
    print(f"\nTraining {horizon_days}-trading-day models")
    training_df = build_training_frame(
        market_df,
        horizon_days=horizon_days,
        gemini_data=gemini_data,
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
        horizon_days,
    )
    risk_model = train_volatility_predictor(
        train_df,
        FEATURE_COLUMNS,
        "future_volatility_outcome",
        horizon_days,
    )

    return_path = save_model(return_model, f"gbr_return_model_{horizon_days}d.pkl")
    risk_path = save_model(risk_model, f"rfr_volatility_model_{horizon_days}d.pkl")

    # Prove the saved files can be loaded again
    loaded_return_model = joblib.load(return_path)
    loaded_risk_model = joblib.load(risk_path)

    prediction_df = build_model_feature_frame(
        market_df,
        gemini_data,
        rolling_volatility_window=ROLLING_VOLATILITY_WINDOW,
        kalman_q=KALMAN_Q,
        kalman_r=KALMAN_R,
    )
    report = latest_prediction_report(
        prediction_df,
        loaded_return_model,
        loaded_risk_model,
        horizon_days=horizon_days,
    )
    print("\nSaved models:")
    print(f"- {return_path}")
    print(f"- {risk_path}")
    print("\nLatest per-ticker predictions:")
    print(report.to_string(index=False))

    return_mae = return_test_mae_percent(test_df, loaded_return_model)
    volatility_mae = volatility_test_mae_percent(test_df, loaded_risk_model)

    return {
        "horizon_days": horizon_days,
        "training_rows": len(training_df),
        "return_test_mae_percent": return_mae,
        "volatility_test_mae_percent": volatility_mae,
        "return_model_path": return_path,
        "risk_model_path": risk_path,
        "gbr_n_estimator": loaded_return_model.get_params()["n_estimators"],
        "gbr_learning": loaded_return_model.get_params()["learning_rate"],
        "gbr_depth": loaded_return_model.get_params()["max_depth"],
        "gbr_subsample": loaded_return_model.get_params()["subsample"],
        "return_half_life_days": return_half_life_days(horizon_days),
        "return_weight_equation": "exp(-(ln(2)/return_half_life_days)*age_days)",
        "volatility_weight_equation": "exp(-(ln(2)/timeline_days)*age_days)",
        "rfr_n_estimator": loaded_risk_model.get_params()["n_estimators"],
        "rfr_depth": loaded_risk_model.get_params()["max_depth"],
        "rfr_samples": loaded_risk_model.get_params()["min_samples_split"],
        "gemini_scoring": GEMINI_SCORING_USED,
        "gemini_min_score": training_df["gemini_sentiment_score"].min(),
        "gemini_max_score": training_df["gemini_sentiment_score"].max(),
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
    }, report


def print_sheet_rows(results: list[dict]) -> None:
    columns = [
        "gbr_n_estimator",
        "gbr_learning",
        "gbr_depth",
        "gbr_subsample",
        "return_half_life_days",
        "return_weight_equation",
        "volatility_weight_equation",
        "rfr_n_estimator",
        "rfr_depth",
        "rfr_samples",
        "gemini_scoring",
        "gemini_min_score",
        "gemini_max_score",
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
            "gemini_min_score": 2,
            "gemini_max_score": 2,
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


def print_prediction_rows(prediction_reports: list[pd.DataFrame]) -> None:
    rows = pd.concat(prediction_reports, ignore_index=True).round(
        {
            "predicted_return": 4,
            "predicted_volatility": 4,
            "gemini_sentiment_score": 2,
        }
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / "latest_predictions.tsv"
    rows.to_csv(output_path, sep="\t", index=False)

    print("\nLatest Predictions TSV:")
    print(rows.to_csv(sep="\t", index=False).strip())
    print(f"\nSaved latest predictions: {output_path}")


async def main_async():
    market_df = fetch_ticker_data(TICKERS, lookback_years = LOOKBACK_YEARS)
    if market_df.empty:
        raise RuntimeError("No market data was downloaded. Check yfinance/network access.")

    latest_market_date = pd.to_datetime(market_df["date"]).max().date().isoformat()
    results = []
    prediction_reports = []
    for horizon_days in PREDICTION_HORIZON_DAYS:
        gemini_data = await fetch_gemini_records(
            TICKERS,
            latest_market_date,
            horizon_days,
        )
        result, prediction_report = train_models_for_horizon(
            market_df,
            gemini_data,
            horizon_days,
        )
        results.append(result)
        prediction_reports.append(prediction_report)

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
    print_prediction_rows(prediction_reports)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
