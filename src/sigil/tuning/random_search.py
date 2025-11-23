from sigil.models import BaseModel
from sigil.distributions import Distribution
from typing import Any, Dict, Sequence, List, Type, Union
import numpy as np


class RandomSearch:
    def __init__(
        self,
        model: Type[BaseModel],
        param_distributions: Dict[
            str, Union[Sequence[Any], Distribution]
        ],  # values can be list (discrete) or tuple (continuous)
        n_iter: int = 10,
        cv: int = 5,
    ) -> None:
        """
        Random Search (Can be discrete or continuous hyperparameters)
        - model = The machine learning model class to be tuned
        - param_distributions = A dictionary where keys are parameter names and values are lists of parameter settings to try
            For discerete parameters, example (Behaves like GridSearch but not exhaustive):
                - "lr" : [0.01, 0.1, 0.2]
                - "n-estimators" : [50, 100, 200]

            For continuous parameters, example:
                - "lr" : (0.0, 0.5)
                - "n-estimators" : (50, 200)

        - n_iter = Number of random combinations to try
        - cv = Number of cross-validation folds
        """
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv

        self.best_params: Dict[str, Any] = {}
        self.best_score = float("-inf")
        self.best_model = None
        self.results: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        keys = list(self.param_distributions.keys())
        if not keys:
            raise ValueError(
                "param_distributions must contain at least one parameter to search over."
            )

        for _ in range(self.n_iter):
            combination = {}
            for key in keys:
                values = self.param_distributions[key]
                val: Union[int, float] = 0

                if isinstance(
                    values, Distribution
                ):  # if distribution object, just call sample method
                    val = values.sample()
                else:  # Fallback to just discrete sampling
                    val = np.random.choice(values)

                # This is to ensure that numpy types are converted to native python types
                if isinstance(val, (np.integer, int)):
                    combination[key] = int(val)
                elif isinstance(val, (np.floating, float)):
                    combination[key] = float(val)

            score = self._cross_validate(X, y, combination)
            self.results.append({"params": combination, "score": score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = combination

        self.best_model = self.model(**self.best_params)
        self.best_model.fit(X, y)

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]
    ) -> float:
        n_samples = len(X)
        fold_size = n_samples // self.cv
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        scores = []
        for i in range(self.cv):
            start = i * fold_size
            end = (i + 1) * fold_size

            idx = indices[start:end]
            train_idx = np.concatenate((indices[:start], indices[end:]))
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[idx], y[idx]

            model = self.model(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            mse = np.mean((y_val - predictions) ** 2)
            scores.append(mse)

        return float(np.mean(scores))
