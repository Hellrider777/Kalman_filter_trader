"""Kalman Filtering state-space execution module."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


class KalmanStateSpaceEngine:
    def __init__(
        self,
        feature_cols: List[str],
        process_noise_cov: float = 1e-4,
        observation_noise_cov: float = 1e-3,
    ):
        self.feature_cols = feature_cols
        self.q_cov = process_noise_cov
        self.r_cov = observation_noise_cov

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        data = df.copy()

        # Reshape X to 3D: (N, 1, n_features)
        X = data[self.feature_cols].values[:, np.newaxis, :]
        y = data["Log_Returns"].values.reshape(-1, 1)

        n_features = X.shape[2]

        initial_state_mean = np.zeros(n_features)
        initial_state_covariance = np.eye(n_features) * 1.0
        transition_matrix = np.eye(n_features)
        transition_covariance = np.eye(n_features) * self.q_cov
        obs_covariance = np.array([[self.r_cov]])

        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=n_features,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_matrices=transition_matrix,
            observation_matrices=X,
            observation_covariance=obs_covariance,
            transition_covariance=transition_covariance,
        )

        state_means, _ = kf.filter(y)

        state_cols = [f"State_{i}" for i in range(n_features)]
        for i, col in enumerate(state_cols):
            data[col] = state_means[:, i]

        X_2d = np.squeeze(X, axis=1)
        predicted_returns_kf = np.sum(X_2d * state_means, axis=1, keepdims=True)
        data["Predicted_Returns_KF"] = predicted_returns_kf
        data["Innovation"] = y - predicted_returns_kf

        return data, state_cols