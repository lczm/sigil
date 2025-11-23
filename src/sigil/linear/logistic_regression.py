from sigil.models import BaseModel
from sigil.activations import sigmoid
from typing import Optional
from tqdm import tqdm
import numpy as np


class LogisticRegression(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=1000, **kwargs) -> None:
    
        if kwargs:
            raise ValueError(f"Unknown parameter(s) for LogisticRegression: {list(kwargs.keys())}")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.n_features: int
        self.n_samples: int

    def build(self, X: np.ndarray, **kwargs) -> None:
        self.n_samples, self.n_features = X.shape

        if "initial_weights" in kwargs:
            self.weights = kwargs["initial_weights"]
        else:
            self.weights = np.zeros(self.n_features)

        if "initial_bias" in kwargs:
            self.bias = kwargs["initial_bias"]
        else:
            self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.weights is None or self.bias is None:
            self.build(X)

        for _ in tqdm(range(self.n_iterations)):
            self.step(X, y)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        predict binary labels (0 or 1) for the given input
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError(
                "Model has not been trained yet. Please call 'fit' before 'predict'."
            )

        # handle multiple dimensions
        # if input_data is 1D, reshape to 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        y_predicted = input_data @ self.weights + self.bias
        y_predicted = sigmoid(y_predicted)

        return (y_predicted >= 0.5).astype(int)

    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.weights is None or self.bias is None or self.n_iterations <= 0:
            raise RuntimeError(
                "Model has not been trained yet. Please call 'fit' before 'step'."
            )

        # logistic regression and linear regression are similar in implementation
        # logistic regression just runs through sigmoid after computing
        y_predicted = X @ self.weights + self.bias
        y_predicted = sigmoid(y_predicted)
        # just like in linear regression, compute gradients
        dw = (1 / self.n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / self.n_samples) * np.sum(y_predicted - y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
