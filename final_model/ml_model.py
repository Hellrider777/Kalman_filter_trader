"""Supervised Machine Learning module with Information Coefficient (IC) evaluation."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class ReturnsPredictor:
    def __init__(
        self,
        forecast_horizon: int = 5,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        max_features: float = 0.5,
        **kwargs,  # Gracefully handle any extra GA parameters
    ):
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state

        # Initialize RandomForestRegressor with GA-optimized hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        # Continuous 5-day forward return target
        data["Target_Ret_5D"] = (
            data["Close"].shift(-self.forecast_horizon) / data["Close"] - 1.0
        )
        return data.dropna(subset=["Target_Ret_5D"]).copy()

    def calculate_ic(self, predictions: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
        """Calculates Pearson IC and Spearman Rank IC."""
        ic, _ = pearsonr(predictions, actuals)
        rank_ic, _ = spearmanr(predictions, actuals)
        return ic, rank_ic

    def train_and_evaluate(
        self, df: pd.DataFrame, feature_cols: List[str], train_split: float = 0.8
    ) -> Tuple[pd.DataFrame, dict]:
        data = df.copy()
        X = data[feature_cols].values
        y = data["Target_Ret_5D"].values

        split_idx = int(len(X) * train_split)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        train_preds = self.model.predict(X_train_scaled)
        test_preds = self.model.predict(X_test_scaled)

        train_ic, train_rank_ic = self.calculate_ic(train_preds, y_train)
        test_ic, test_rank_ic = self.calculate_ic(test_preds, y_test)

        metrics = {
            "train_ic": train_ic,
            "test_ic": test_ic,
            "train_rank_ic": train_rank_ic,
            "test_rank_ic": test_rank_ic,
        }
        importances = pd.Series(self.model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\n🔥 Top 5 Predictive Features:")
        print(importances.head(5).to_string())

        X_all_scaled = self.scaler.transform(X)
        data["Predicted_Forward_Return"] = self.model.predict(X_all_scaled)

        return data, metrics