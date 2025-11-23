import numpy as np
from .distributions import Distribution
from typing import Union


class Uniform(Distribution):
    def __init__(self, low: Union[int, float], high: Union[int, float]) -> None:
        """
        Initialize the Uniform distribution with a range of values.
        """

        if low >= high:
            raise ValueError("Low value must be less than high value.")

        self.low = low
        self.high = high

    def sample(self) -> float:
        """
        Sample from the Uniform distribution.
        """
        return np.random.uniform(self.low, self.high)
