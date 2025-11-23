from sigil.models import BaseModel
from typing import Any, Dict, List, Type
import itertools
import numpy as np


class GridSearch:
    def __init__(
        self,
        model: Type[BaseModel],
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
    ) -> None:
        """
        Initialize the GridSearch with a model, parameter grid, and number of cross-validation folds.
        - model = The machine learning model class to be tuned
        - param_grid = A dictionary where keys are parameter names and values are lists of parameter settings to try
        - cv = Number of cross-validation folds
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv

        self.best_params: Dict[str, Any] = {}
        self.best_score = float("-inf")
        self.best_model = None
        self.results: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if not combinations:
            raise ValueError(
                "param_grid must contain at least one parameter to search over."
            )

        for combination in combinations:
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
