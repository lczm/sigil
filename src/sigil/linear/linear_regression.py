import numpy as np


class LinearRegression():
    """
    Implementation of Linear Regression using Gradient Descent.

    Basic idea: y = wx + b
        
    Attributes:
        - bias = y-intercept of the regression line
        - weights = A vector of all coefficients
            So basically, in a 3 dimensional space, weights = [w1, w2, w3]
            y = w1*x1 + w2*x2 + w3*x3 + bias
            This can extend to n-dimensions
    Methods:
        - fit(X, y) = trains the model using the training data
        - predict(X) = predicts the target values for given input data
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000) -> None:
        """
        Initialize the Linear Regression model with learning rate and number of iterations.
        - learning_rate = How fast the model learns
        - n_iterations = Number of times the model will iterate over the training data
        """

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Linear Regression model using Gradient Descent.
        - X = Input features (n dimension structure)
        - y = Target values (1 dimension structure)
        """

        # Samples are the number of data points
        # Features are the number of dimensions in each data point
        n_samples, n_features = X.shape

        # Intial weight and bias should be zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # We can then compute using gradient descent.
        # The basic idea is to minimise the cost function (MSE here)
        # By "rotating" the linear regression line, we can calculate if the MSE is reduced.
        # Eventually, we will reach a point of convergence where the cost function is minimised
        # Using gradient descent minimises the computational cost of finding the optimal weights and bias
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            # Basically, it derives based on MSE formula to find the gradient change with respect to weights and bias
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained Linear Regression model.
        - X = Input features (n dimension structure)
        Returns predicted target values.
        """
        # The number of features in X should match the number of weights
        if X.shape[1] != len(self.weights):
            raise ValueError("Number of features in input data must match number of weights.")

        # Basically it does this
        # y_predicted = w1*x1 + w2*x2 + ... + wn*xn + bias
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted




