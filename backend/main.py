"""
This file contains the program to output a file containing optimally-weighted composition for an ETF through a CLI, tailored to user inputs.
Run from backend/ with virtual env activated (with dependencies installed):

python main.py \
  --file_name {str} \
  --horizon_days {int} \
  --sector {str1}={float1} {str2}={float2} ... \
  --size {str1} {str2} ... \
  --exclude {str1} {str2} ... \
  --min_pool {int} \
  --max_pool {int} \
  --risk_tolerance {str}

Note: at least one argument is needed for each argument type except for "exclude", additional is optional. Do not zero-out excluded sectors
"""

import argparse
from collections import defaultdict
import math
from pathlib import Path
import matplotlib.pyplot as plt


RESULTS_PATH = Path(__file__).resolve().parent / "etf_output_files"


def return_optimized_etf(ml_inputs: dict, risk_tolerance: str) -> dict:
    """To be implemented in math_engine/"""
    pass

def shape_user_inputs(args: argparse.Namespace) -> dict:
    """Shapes the user inputs for ml ticker collection and inference, and a component in ETF composition function."""
    live_inference_inputs = {}

    live_inference_inputs["horizon_days"] = args.horizon_days
    sect_mapping = {}
    for s in list(args.sector):
        sect, cap = s.split("=", 1)
        sect_mapping[sect] = float(cap)
    live_inference_inputs["sectors"] = sect_mapping
    live_inference_inputs["sizes"] = list(args.size)
    live_inference_inputs["blacklisted"] = list(args.exclude) or []
    live_inference_inputs["min_pool"] = int(args.min_pool)
    live_inference_inputs["max_pool"] = int(args.max_pool)

    return live_inference_inputs

def output_results_file(results: dict[str, dict], path: Path) -> None:
    """Save a portfolio report from ticker to prediction and composition records.

    Each record must contain weight, return, volatility, sector, and market_cap_category. 
    Weights are decimal portfolio weights and must sum to 1.0.
    """
    required_fields = {
        "weight",
        "return",
        "volatility",
        "sector",
        "market_cap_category",
    }
    if not results:
        raise ValueError("No portfolio was created")

    missing_fields = {
        ticker: required_fields - record.keys()
        for ticker, record in results.items()
        if required_fields - record.keys()
    }
    if missing_fields:
        raise ValueError(f"Portfolio records are missing fields: {missing_fields}")

    holdings = [
        (ticker, record)
        for ticker, record in results.items()
        if record["weight"] > 0
    ]
    total_weight = sum(record["weight"] for _, record in holdings)
    if not math.isclose(total_weight, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Portfolio weights must sum to 1.0")

    def composition(field: str) -> dict[str, float]:
        result = defaultdict(float)
        for _, record in holdings:
            result[record[field]] += record["weight"]
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    def add_pie_and_summary(pie_axis, summary_axis, title: str, values: dict[str, float]) -> None:
        labels = list(values)
        weights = list(values.values())
        pie_axis.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90)
        pie_axis.set_title(title)

        summary_axis.axis("off")
        summary_axis.table(
            cellText=[[label, f"{weight:.2%}"] for label, weight in values.items()],
            colLabels=["Holding", "Weight"],
            cellLoc="left",
            loc="center",
        )

    ticker_composition = dict(
        sorted(
            ((ticker, record["weight"]) for ticker, record in holdings),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    sector_composition = composition("sector")
    market_cap_composition = composition("market_cap_category")

    figure = plt.figure(figsize=(15, 16), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=[4, 4, 4, 1])
    figure.suptitle("Optimized ETF Composition", fontsize=16, fontweight="bold")

    add_pie_and_summary(
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        "ETF Ticker Composition",
        ticker_composition,
    )
    add_pie_and_summary(
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        "ETF Composition by Sector",
        sector_composition,
    )
    add_pie_and_summary(
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
        "ETF Composition by Market Cap",
        market_cap_composition,
    )

    expected_return = sum(record["weight"] * record["return"] for _, record in holdings)
    expected_volatility = sum(record["weight"] * record["volatility"] for _, record in holdings)
    metrics_axis = figure.add_subplot(grid[3, :])
    metrics_axis.axis("off")
    metrics_axis.text(
        0.5,
        0.5,
        f"Expected return: {expected_return:.2%}    |    Expected volatility: {expected_volatility:.2%}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(
        description = "Return a tailored optimized ETF portfolio"
    )

    parser.add_argument("--file_name", type = str, required = True)
    parser.add_argument("--horizon_days", type = int, required = True)
    parser.add_argument("--sector", type = str, nargs = "+", required = True)
    parser.add_argument("--size", type = str, nargs = "+", required = True)
    parser.add_argument("--exclude", type = str, nargs = "+", default = [])
    parser.add_argument("--min_pool", type = int, required = True)
    parser.add_argument("--max_pool", type = int, required = True)
    parser.add_argument("--risk_tolerance", type = str, required = True)

    args = parser.parse_args()

    ml_inputs = shape_user_inputs(args)

    results = return_optimized_etf(ml_inputs, args.risk_tolerance)

    filepath = RESULTS_PATH / f"{args.file_name}.png"
    output_results_file(results, filepath)
