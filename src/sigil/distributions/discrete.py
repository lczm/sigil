import random
from .distributions import Distribution    
from typing import Union, List

class Discrete(Distribution):
    def __init__(self, values: List[Union[int, float]]) -> None:
        """
        Initialize the Discrete distribution with a list of probabilities.
        """
        self.values = values
    
    def sample(self) -> Union[int, float]:
        """
        Sample from the discrete distribution.
        """
        return random.choice(self.values)

