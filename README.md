# 📈 Quantitative Macro Trading Engine (Kalman Filter + GA + ML)

A high-performance, systematic quantitative trading engine designed to generate alpha on multi-asset equities using **State-Space Kalman Filtering**, **Genetic Algorithm Hyperparameter Optimization**, and **Information Coefficient (IC) ML Ranking Regressors**.

---

## 🛠️ Key Architectural Features

1. **State-Space Kalman Filter Engine:**
   * Isolates dynamic underlying asset trends (`State_1`) and velocity/momentum (`State_2`) by filtering out high-frequency market microstructure noise.
2. **Genetic Algorithm (GA) Hyperparameter Optimization:**
   * Automates hyperparameter search (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) to continuously adapt model structure to shifting market regimes without overfitting.
3. **Information Coefficient (IC) ML Ranking Model:**
   * Uses continuous 5-day forward return predictions (`Target_Ret_5D`) optimized for **Spearman Rank Correlation (Rank IC)** rather than brittle binary directional classification.
4. **Zero-Leakage Execution Backtest Engine:**
   * Features strict $t+1$ bar execution, dynamic volatility stop-losses via ATR, time-based exits, and realistic transaction cost/slippage modeling ($10 \text{ bps}$).

---

## 📊 Backtest Performance (Out-of-Sample)

Tested on 10-year historical market data ($2,515$ market bars) for **MSFT** using macro and cross-asset context (**QQQ**, **SPY**, **^VIX**):

| Metric | Strategy Value | Benchmark (Buy & Hold) |
| :--- | :--- | :--- |
| **Total Return** | **`2,627.16%`** | `1,132.55%` |
| **Sharpe Ratio** | **`1.593`** | ~`0.85` |
| **Max Drawdown** | **`-22.20%`** | `-33.75%` |
| **Out-of-Sample Pearson IC** | **`0.0635`** | N/A |
| **Out-of-Sample Rank IC** | **`0.0787`** | N/A |

> **Alpha Signal:** An out-of-sample Rank IC of **`0.0787`** (~$7.9\%$) indicates high predictive rank-ordering power for forward equity returns.

---

## 🔥 Top Predictive Features

Feature importance diagnostics from the GA-optimized Random Forest model:

1. **`State_1`** (Kalman Trend Extraction)
2. **`VIX_Level`** (Macro Volatility & Regime Filter)
3. **`MA_Ratio_5_20`** (Short-Term Trend Acceleration)
4. **`State_2`** (Kalman Velocity/Momentum)
5. **`ATR_14`** (Volatilty Expansion Meter)

---

## ⚙️ Quick Start & Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone [https://github.com/Hellrider777/Kalman_filter_trader.git](https://github.com/Hellrider777/Kalman_filter_trader.git)
cd Kalman_filter_trader

python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt