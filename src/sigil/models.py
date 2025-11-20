from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def build(self, X: np.ndarray, **kwargs) -> None:
        """
        Method to initialize internal model state based on input shape
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Method to train the regression model.
        """
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Method to predict target values for given input data.
        """
        pass

    @abstractmethod
    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Method to perform an iterative step in the training process.
        """
        pass
