"""Configuration settings including Macro and Sector tickers."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyConfig:
    # Primary Asset & Macro Benchmarks
    ticker: str = "MSFT"
    macro_tickers: List[str] = field(
        default_factory=lambda: ["QQQ", "SPY", "^VIX"]
    )
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    random_seed: int = 42

    # Prediction Horizon
    forecast_horizon: int = 5            # 5-day forward target

    # Kalman Filter Parameters
    kalman_features: List[str] = field(
        default_factory=lambda: [
            "MA_Ratio_5_20",
            "Returns_Lag_1",
            "Returns_Lag_5",
            "MSFT_vs_QQQ_5D",
            "VIX_Change_5D",
            "RSI",
        ]
    )
    process_noise_cov: float = 1e-4
    observation_noise_cov: float = 1e-3

    # Machine Learning Settings
    train_split_ratio: float = 0.8
    probability_threshold: float = 0.51 

    # Risk & Strategy Execution
    stop_loss_mult: float = 2.5
    max_holding_period: int = 5

    # Backtest Configuration
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02