from abc import ABC, abstractmethod
from typing import Union

class Distribution(ABC):

    @abstractmethod
    def sample(self) -> Union[int, float]:
        """
        Method to generate samples from the distribution.
        """
        pass
