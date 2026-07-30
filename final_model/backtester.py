"""Backtesting engine with continuous return prediction signals."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class BacktestEngine:
    def __init__(
        self,
        entry_threshold: float = 0.005,   # Long entry: Expecting > +0.5% return over 5 days
        exit_threshold: float = -0.002,   # Exit/Short: Expecting < -0.2% return
        stop_loss_mult: float = 2.0,
        max_holding_period: int = 15,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_mult = stop_loss_mult
        self.max_holding_period = max_holding_period
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rf_daily = risk_free_rate / 252.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates trade signals strictly based on Predicted_Forward_Return values."""
        data = df.copy()
        data["Signal"] = 0

        current_position = 0
        entry_price = 0.0
        days_held = 0

        for i in range(len(data)):
            # Switched from Prob_Up_5D to Predicted_Forward_Return
            pred_return = data["Predicted_Forward_Return"].iloc[i]
            current_price = data["Close"].iloc[i]
            atr = data["ATR_14"].iloc[i]

            if current_position == 0:
                # Long Entry Condition (> +0.5% expected return)
                if pred_return > self.entry_threshold:
                    current_position = 1
                    entry_price = current_price
                    days_held = 0
                    data.iloc[i, data.columns.get_loc("Signal")] = 1
                # Short Entry Condition (< -0.2% expected return)
                elif pred_return < self.exit_threshold:
                    current_position = -1
                    entry_price = current_price
                    days_held = 0
                    data.iloc[i, data.columns.get_loc("Signal")] = -1
            else:
                days_held += 1
                pnl = (
                    (current_price - entry_price)
                    if current_position == 1
                    else (entry_price - current_price)
                )

                exit_triggered = False

                # Risk / Exit Management Rules
                if pnl < -(self.stop_loss_mult * atr):
                    exit_triggered = True
                elif (current_position == 1 and pred_return < 0.0) or (
                    current_position == -1 and pred_return > 0.0
                ):
                    exit_triggered = True  # Signal reversed
                elif days_held >= self.max_holding_period:
                    exit_triggered = True  # Time stop

                if exit_triggered:
                    data.iloc[i, data.columns.get_loc("Signal")] = -current_position
                    current_position = 0
                    entry_price = 0.0
                    days_held = 0

        return data

    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Executes trades on day t+1 using signals generated at day t close."""
        data = self.generate_signals(df)

        # Shift signals by 1 bar to ensure t+1 execution
        data["Exec_Signal"] = data["Signal"].shift(1).fillna(0)

        n = len(data)
        cash = np.zeros(n)
        holdings = np.zeros(n)
        portfolio_value = np.zeros(n)

        current_cash = float(self.initial_capital)
        current_holdings = 0.0

        for i in range(n):
            exec_signal = data["Exec_Signal"].iloc[i]
            price = data["Close"].iloc[i]

            # Execute pending signals
            if exec_signal == 1 and current_holdings == 0:  # Enter Long
                shares = int((current_cash * 0.95) / price)
                if shares > 0:
                    cost = shares * price
                    fee = cost * self.transaction_cost
                    if (cost + fee) <= current_cash:
                        current_cash -= (cost + fee)
                        current_holdings += shares

            elif exec_signal == -1 and current_holdings > 0:  # Exit Long
                revenue = current_holdings * price
                fee = revenue * self.transaction_cost
                current_cash += (revenue - fee)
                current_holdings = 0.0

            cash[i] = current_cash
            holdings[i] = current_holdings
            portfolio_value[i] = current_cash + (current_holdings * price)

        data["Cash"] = cash
        data["Holdings"] = holdings
        data["Portfolio_Value"] = portfolio_value

        # Metrics computation
        data["Strategy_Returns"] = data["Portfolio_Value"].pct_change().fillna(0.0)
        data["Cumulative_Returns"] = (1 + data["Strategy_Returns"]).cumprod() - 1

        data["BuyHold_Returns"] = data["Close"].pct_change().fillna(0.0)
        data["BuyHold_Cumulative"] = (1 + data["BuyHold_Returns"]).cumprod() - 1

        final_val = data["Portfolio_Value"].iloc[-1]
        total_ret = (final_val - self.initial_capital) / self.initial_capital

        excess_rets = data["Strategy_Returns"] - self.rf_daily
        sharpe = (
            np.sqrt(252) * excess_rets.mean() / excess_rets.std()
            if excess_rets.std() != 0
            else 0.0
        )

        cum_series = (1 + data["Strategy_Returns"]).cumprod()
        drawdown = (cum_series - cum_series.expanding().max()) / cum_series.expanding().max()
        max_dd = drawdown.min()

        metrics = {
            "Final_Value": final_val,
            "Total_Return": total_ret,
            "Sharpe_Ratio": sharpe,
            "Max_Drawdown": max_dd,
            "Benchmark_Return": data["BuyHold_Cumulative"].iloc[-1],
        }

        return data, metrics