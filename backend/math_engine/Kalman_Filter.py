import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


class Kalman_Filter:
    def __init__(self, Q=1e-5, R=1e-2):
        self.Q = Q
        self.R = R

    def filter(self, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[float(prices.iloc[0])], [0.0]])
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.P *= 1.0
        kf.R = np.array([[self.R]])
        kf.Q = np.eye(2) * self.Q

        smoothed_prices, smoothed_velocities = [], []
        for price in prices:
            kf.predict()
            kf.update(np.array([[float(price)]]))
            smoothed_prices.append(kf.x[0, 0])
            smoothed_velocities.append(kf.x[1, 0])

        return (
            pd.Series(smoothed_prices, index=prices.index),
            pd.Series(smoothed_velocities, index=prices.index),
        )

    def smooth_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts the DataFrame from market_data_collection.fetch_ticker_data()
        and attaches kalman_smoothed_price and kalman_velocity columns.
        Processes each ticker independently since they're interleaved in the DF.
        """
        result = df.copy()
        result['kalman_smoothed_price'] = np.nan
        result['kalman_velocity'] = np.nan

        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            prices = df.loc[mask, 'adjusted_close']

            smoothed, velocity = self.filter(prices)

            result.loc[mask, 'kalman_smoothed_price'] = smoothed.values
            result.loc[mask, 'kalman_velocity'] = velocity.values

        return result