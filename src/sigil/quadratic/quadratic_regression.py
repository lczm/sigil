from sigil.models import BaseRegressionModel
import numpy as np

class QuadraticRegression(BaseRegressionModel):
    """
    Implementation of Quadratic Regression using Gradient Descent.
    Basic idea: y = ax^2 + bx + c
    Principles are the same as linear regression, but we introduce squared terms to capture non-linear relationships.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000) -> None:
        """
        Initialize the Quadratic Regression model with learning rate and number of iterations.
        - learning_rate = How fast the model learns
        - n_iterations = Number of times the model will iterate over the training data
        
        The mean and std are stored to apply the same scaling in the predict method when used in the fit method
        """

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

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
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
        X_poly = np.column_stack((X_scaled, X_scaled**2))

        n_samples, n_features = X_poly.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # The principle is the same as linear regression, but we work with the expanded feature set
        #  e
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X_poly, self.weights) + self.bias
            dw = (2 / n_samples) * np.dot(X_poly.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Quadratic Regression model.
        - input_data = Input features (n dimension structure)
        Returns predicted target values.
        """

        # If model is not trained, raise error
        if self.weights is None:
            raise RuntimeError("Model has not been trained yet. Please call 'fit' before 'predict'.")

        # Scale input data using the mean and std from training
        input_data_scaled = (input_data - self.mean) / self.std

        # Expand features to include squared terms
        input_data_quad = np.column_stack((input_data_scaled, input_data_scaled**2))

        if input_data_quad.shape[1] != len(self.weights):
            raise ValueError("Number of features in input data must match number of weights.")

        return np.dot(input_data_quad, self.weights) + self.bias
