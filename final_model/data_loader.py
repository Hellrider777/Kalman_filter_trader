"""Data ingestion module for primary asset and macro market drivers."""

import logging
from typing import List
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataLoader:
    def __init__(self, ticker: str, macro_tickers: List[str], start_date: str, end_date: str):
        self.ticker = ticker
        self.macro_tickers = macro_tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        all_symbols = [self.ticker] + self.macro_tickers
        logger.info(f"Downloading historical market data for {all_symbols}...")

        data = yf.download(all_symbols, start=self.start_date, end=self.end_date, progress=False)

        if data.empty:
            raise ValueError("Failed to download market data from Yahoo Finance.")

        # Extract Close prices
        close_df = data["Close"].copy()
        
        # Build unified DataFrame
        df = pd.DataFrame(index=close_df.index)
        df["Close"] = close_df[self.ticker]
        df["High"] = data["High"][self.ticker]
        df["Low"] = data["Low"][self.ticker]
        df["Volume"] = data["Volume"][self.ticker]

        for symbol in self.macro_tickers:
            clean_name = symbol.replace("^", "")
            df[f"Close_{clean_name}"] = close_df[symbol]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.ffill().dropna()
        logger.info(f"Fetched and aligned {len(df)} market bars.")
        return df