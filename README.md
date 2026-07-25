# ETF Optimizer

The ETF Optimizer is an in-progress portfolio researchproject. It combines historical market data, LLM-derived news and fundamentals signals, and numerical machine-learning forecasts to explore stock candidates and support future portfolio-optimization workflows.

It is a research and development tool—not investment advice or an automated trading system.

## Current state

The machine-learning sandbox is the latest milestone achieved in the project. Currently, it can:

- train return and volatility models for 3, 20, and 90-day horizons;
- backtest those models by chronologically separating historical data for instant prediction performance;
- screen US large-cap stocks by sector, collect live market and company inputs, route to and extract responses Gemini instances, and produce return and volatility estimates asynchronously.

For the return Gradient Boosting Regressor and volatility Random Forest Regressor performance, the 20-day prediction horizon model achieved:

- 58.281% mean return directional accuracy (win-rate or correct-direction-rate);
- 0.6583% volatility RMSE (measure of volatility prediction accuracy);
over a 477 test sample size using a mix of tech, finance, and energy sector tickers.

The broader backend and portfolio-optimization layers are still under active development.

## Current setup for testing ML engine

From `backend/`, activate the project virtual environment and choose one workflow in `ml_engine/sandbox_ml_models.py`:

```bash
source ETF_Opt/bin/activate
python -m ml_engine.sandbox_ml_models
```

For example, enabling `backtesting_inference` prints a three-row summary for the 3, 20, and 90-day models, including mean return directional accuracy and volatility RMSE. Enabling `live_inference` screens the configured ticker pool and prints a prediction for each selected ticker.

Live inference requires valid Google Vertex AI authentication and the environment settings in `backend/.flaskenv`.

## Project layout

```text
backend/
  ml_engine/      Training, backtesting, live inference, and data collection
  math_engine/    Kalman price-smoothing utilities
  requirements.txt
```

For backend ownership boundaries, data contracts, and a more detailed module guide, see the [backend README](backend/README.md).

## Intended use

The project is used to personally experiment with candidate selection and model evaluation, inspect forecast outputs, and iterate on portfolio-research ideas.
