"""Main execution pipeline with Macro, Sector, and GA Hyperparameter Optimization."""

import logging
import matplotlib.pyplot as plt

from backtester import BacktestEngine
from config import StrategyConfig
from data_loader import MarketDataLoader
from features import FeatureEngineer
from ga_optimizer import GAHyperparameterOptimizer
from kalman_engine import KalmanStateSpaceEngine
from ml_model import ReturnsPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = StrategyConfig()

    # 1. Fetch Multi-Asset Market Data
    loader = MarketDataLoader(
        ticker=config.ticker,
        macro_tickers=config.macro_tickers,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    raw_df = loader.fetch_data()

    # 2. Build Multi-Asset Features
    df_features = FeatureEngineer.build_features(raw_df)

    # 3. Kalman Filter Analysis
    kalman_engine = KalmanStateSpaceEngine(
        feature_cols=config.kalman_features,
        process_noise_cov=config.process_noise_cov,
        observation_noise_cov=config.observation_noise_cov,
    )
    df_kalman, state_cols = kalman_engine.fit_transform(df_features)

    # 4. Prepare Machine Learning Features with Macro Drivers
    ml_features = state_cols + [
        "Returns_Lag_1",
        "Returns_Lag_5",
        "MSFT_vs_QQQ_1D",
        "MSFT_vs_QQQ_5D",
        "MSFT_vs_SPY_5D",
        "QQQ_Returns_5D",
        "SPY_Returns_5D",
        "VIX_Level",
        "VIX_Change_5D",
        "MA_Ratio_5_20",
        "Volatility_20",
        "ATR_14",
        "RSI",
        "Predicted_Returns_KF",
        "Innovation",
    ]

    # Temporary instance to construct target variable
    temp_predictor = ReturnsPredictor(forecast_horizon=config.forecast_horizon)
    df_target = temp_predictor.prepare_target(df_kalman)

    # 5. Run Genetic Algorithm to find optimal Random Forest parameters
    logger.info("Running Genetic Algorithm hyperparameter optimization...")
    ga_opt = GAHyperparameterOptimizer(population_size=15, generations=5)
    best_params = ga_opt.optimize(df_target, feature_cols=ml_features, target_col="Target_Ret_5D")

    # 6. Initialize & Train Model with GA Best Parameters
    predictor = ReturnsPredictor(
        forecast_horizon=config.forecast_horizon,
        random_state=config.random_seed,
        **best_params,  # Unpacks GA optimal parameters
    )
    
    df_ml, ml_metrics = predictor.train_and_evaluate(
        df_target, feature_cols=ml_features, train_split=config.train_split_ratio
    )

    logger.info(
        f"5-Day Macro IC: Train={ml_metrics['train_ic']:.4f} | Out-of-Sample Test={ml_metrics['test_ic']:.4f}"
    )
    logger.info(
        f"5-Day Macro Rank IC: Train={ml_metrics['train_rank_ic']:.4f} | Out-of-Sample Test={ml_metrics['test_rank_ic']:.4f}"
    )

    # 7. Execute Strategy Backtest
    backtester = BacktestEngine(
        entry_threshold=0.002,   # Lower threshold to stay in strong trends longer (+0.2%)
        exit_threshold=-0.001,   # Exit slightly faster on negative expectations (-0.1%)
        stop_loss_mult=2.5,      # Slightly wider volatility stop to prevent getting shaken out
        max_holding_period=10,
        initial_capital=config.initial_capital,
        transaction_cost=config.transaction_cost,
        risk_free_rate=config.risk_free_rate,
    )

    results_df, stats = backtester.run_backtest(df_ml)

    print("\n" + "=" * 60)
    print(f"MACRO & SECTOR ENHANCED STRATEGY EVALUATION [{config.ticker}]")
    print("=" * 60)
    print(f"Final Portfolio Value: ${stats['Final_Value']:,.2f}")
    print(f"Total Return:          {stats['Total_Return']*100:.2f}%")
    print(f"Benchmark Return:      {stats['Benchmark_Return']*100:.2f}%")
    print(f"Sharpe Ratio:          {stats['Sharpe_Ratio']:.3f}")
    print(f"Max Drawdown:          {stats['Max_Drawdown']*100:.2f}%")
    print("=" * 60 + "\n")

    # 8. Plot Results
    plt.figure(figsize=(12, 5))
    plt.plot(
        results_df.index,
        results_df["Cumulative_Returns"] * 100,
        label="Macro Strategy (%)",
        linewidth=1.8,
    )
    plt.plot(
        results_df.index,
        results_df["BuyHold_Cumulative"] * 100,
        label="Buy & Hold (%)",
        linestyle="--",
        color="gray",
    )
    plt.title(f"{config.ticker} Macro Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()