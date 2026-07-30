"""Feature engineering module with relative strength and macro features."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Primary Stock Returns
        data["Returns"] = data["Close"].pct_change()
        data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Returns_5D"] = data["Close"].pct_change(5)

        # Macro Context Returns
        data["QQQ_Returns_1D"] = data["Close_QQQ"].pct_change(1)
        data["QQQ_Returns_5D"] = data["Close_QQQ"].pct_change(5)
        data["SPY_Returns_1D"] = data["Close_SPY"].pct_change(1)
        data["SPY_Returns_5D"] = data["Close_SPY"].pct_change(5)

        # Volatility Sentiment Features
        data["VIX_Level"] = data["Close_VIX"]
        data["VIX_Change_1D"] = data["Close_VIX"].pct_change(1)
        data["VIX_Change_5D"] = data["Close_VIX"].pct_change(5)

        # Relative Strength Indicators (Alpha vs Sector & Market)
        data["MSFT_vs_QQQ_1D"] = data["Returns"] - data["QQQ_Returns_1D"]
        data["MSFT_vs_QQQ_5D"] = data["Returns_5D"] - data["QQQ_Returns_5D"]
        data["MSFT_vs_SPY_1D"] = data["Returns"] - data["SPY_Returns_1D"]
        data["MSFT_vs_SPY_5D"] = data["Returns_5D"] - data["SPY_Returns_5D"]

        # Moving Averages & Trend Ratios
        data["MA_5"] = data["Close"].rolling(5).mean()
        data["MA_20"] = data["Close"].rolling(20).mean()
        data["MA_60"] = data["Close"].rolling(60).mean()

        data["MA_Ratio_5_20"] = data["MA_5"] / data["MA_20"]
        data["MA_Ratio_20_60"] = data["MA_20"] / data["MA_60"]

        # Lags & Rate of Change
        for lag in [1, 2, 5]:
            data[f"Returns_Lag_{lag}"] = data["Returns"].shift(lag)

        data["ROC_5"] = ((data["Close"] - data["Close"].shift(5)) / data["Close"].shift(5)) * 100
        data["ROC_10"] = ((data["Close"] - data["Close"].shift(10)) / data["Close"].shift(10)) * 100

        # Technical Indicators (ATR, Volatility, RSI)
        data["Volatility_20"] = data["Returns"].rolling(20).std()
        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Close"].shift(1)).abs()
        low_close = (data["Low"] - data["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["ATR_14"] = tr.rolling(14).mean()

        data["Volume_MA_20"] = data["Volume"].rolling(20).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_MA_20"]

        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        return data.dropna().copy()