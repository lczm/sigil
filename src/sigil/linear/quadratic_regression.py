from sigil.models import BaseModel
from tqdm import tqdm
from typing import Optional
import numpy as np


class QuadraticRegression(BaseModel):
    """
    Implementation of Quadratic Regression using Gradient Descent.
    Basic idea: y = ax^2 + bx + c
    Principles are the same as linear regression, but we introduce squared terms to capture non-linear relationships.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, **kwargs) -> None:
        """
        Initialize the Quadratic Regression model with learning rate and number of iterations.
        - learning_rate = How fast the model learns
        - n_iterations = Number of times the model will iterate over the training data
        - n_features = Features are the number of dimensions in each data point
        - n_samples = Samples are the number of data points


        The mean and std are stored to apply the same scaling in the predict method when used in the fit method
        """
        if kwargs:
            raise ValueError(f"Unknown parameter(s) for QuadraticRegression: {list(kwargs.keys())}")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.n_features: int
        self.n_samples: int
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def build(self, X: np.ndarray, **kwargs) -> None:
        self.n_samples, self.n_features = X.shape

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # Since we are adding squared terms, we need to double the number of features
        self.n_features *= 2

        if "initial_weights" in kwargs:
            self.weights = kwargs["initial_weights"]
        else:
            self.weights = np.zeros(self.n_features)

        if "initial_bias" in kwargs:
            self.bias = kwargs["initial_bias"]
        else:
            self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Quadratic Regression model using Gradient Descent.
        - X = Input features (n dimension structure)
        - y = Target values (1 dimension structure)
        """
        # Add squared features: [x, x²]
        # Essentially, if is the original set is
        # [ 1 ]
        # [ 2 ]
        # [ 3 ]
        # The new set will be
        # [ 1, 1 ]
        # [ 2, 4 ]
        # [ 3, 9 ]
        # But since squared terms grows fast, we will normalise them to avoid computational issues

        # X - X.mean(axis=0) computes mean of the feature.
        # Example, [2,4,6]
        # Mean = (2+4+6)/3 = 4
        # X - X.mean(axis=0) = [-2, 0, 2]
        # Dividing by std dev scales it to between [-1, 1], this prevents the values from being too large.
        if self.weights is None or self.bias is None:
            self.build(X)

        X_scaled = (X - self.mean) / self.std
        X_poly = np.column_stack((X_scaled, X_scaled**2))

        # The principle is the same as linear regression, but we work with the expanded feature set
        for _ in tqdm(range(self.n_iterations)):
            self.step(X_poly, y)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Quadratic Regression model.
        - input_data = Input features (n dimension structure)
        Returns predicted target values.
        """

        # If model is not trained, raise error
        if self.weights is None or self.bias is None:
            raise RuntimeError(
                "Model has not been trained yet. Please call 'fit' before 'predict'."
            )

        # Scale input data using the mean and std from training
        input_data_scaled = (input_data - self.mean) / self.std

        # Expand features to include squared terms
        input_data_quad = np.column_stack((input_data_scaled, input_data_scaled**2))

        if input_data_quad.shape[1] != len(self.weights):
            raise ValueError(
                "Number of features in input data must match number of weights."
            )

        return np.dot(input_data_quad, self.weights) + self.bias

    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.weights is None or self.bias is None or self.n_iterations <= 0:
            raise RuntimeError(
                "Model has not been trained yet. Please call 'fit' before 'step'."
            )

        y_predicted = np.dot(X, self.weights) + self.bias
        dw = (2 / self.n_samples) * np.dot(X.T, (y_predicted - y))
        db = (2 / self.n_samples) * np.sum(y_predicted - y)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
