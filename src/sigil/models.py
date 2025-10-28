from abc import ABC, abstractmethod
import numpy as np

class BaseRegressionModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Method to train the regression model."""
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Method to predict target values for given input data."""
        pass
