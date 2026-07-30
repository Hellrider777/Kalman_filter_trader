"""Genetic Algorithm hyperparameter optimizer for Random Forest Regressor."""

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr


class GAHyperparameterOptimizer:
    def __init__(self, population_size: int = 15, generations: int = 5):
        self.pop_size = population_size
        self.generations = generations

    def _eval_individual(self, individual, X_train, y_train, X_val, y_val):
        """Fitness function: Returns Out-of-Sample Rank IC on validation set."""
        n_estimators = int(individual[0])
        max_depth = int(individual[1])
        min_samples_leaf = int(individual[2])
        max_features = float(individual[3])

        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            # Evaluate on validation set using Spearman Rank Correlation
            val_preds = model.predict(X_val)
            rank_ic, _ = spearmanr(val_preds, y_val)

            if np.isnan(rank_ic):
                return (0.0,)
            return (rank_ic,)
        except Exception:
            return (-1.0,)

    def optimize(
        self, df: pd.DataFrame, feature_cols: list, target_col: str = "Target_Ret_5D"
    ) -> dict:
        """Runs the GA to search for optimal hyperparameters."""
        data = df.dropna().copy()
        X = data[feature_cols].values
        y = data[target_col].values

        # Split data: 60% Train, 20% Validation (for GA Fitness), 20% Held-out Test
        train_idx = int(len(X) * 0.6)
        val_idx = int(len(X) * 0.8)

        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]

        # Setup DEAP Genetic Algorithm Structures
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Genes: [n_estimators, max_depth, min_samples_leaf, max_features]
        toolbox.register("n_estimators", random.randint, 50, 300)
        toolbox.register("max_depth", random.randint, 2, 6)
        toolbox.register("min_samples_leaf", random.randint, 20, 100)
        toolbox.register("max_features", random.uniform, 0.2, 0.8)

        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (
                toolbox.n_estimators,
                toolbox.max_depth,
                toolbox.min_samples_leaf,
                toolbox.max_features,
            ),
            n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register(
            "evaluate",
            self._eval_individual,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=[50, 2, 20, 0],
            up=[300, 6, 100, 1],
            indpb=0.2,
        )
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.pop_size)

        # Run Evolution Loop
        algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=0.5,
            mutpb=0.2,
            ngen=self.generations,
            verbose=False,
        )

        best_ind = tools.selBest(pop, 1)[0]

        best_params = {
            "n_estimators": int(best_ind[0]),
            "max_depth": int(best_ind[1]),
            "min_samples_leaf": int(best_ind[2]),
            "max_features": round(float(best_ind[3]), 2),
        }

        print(f"\n🧬 [GA Optimization Complete] Best Params Found:")
        print(f"   • n_estimators:     {best_params['n_estimators']}")
        print(f"   • max_depth:        {best_params['max_depth']}")
        print(f"   • min_samples_leaf: {best_params['min_samples_leaf']}")
        print(f"   • max_features:     {best_params['max_features']}\n")

        return best_params