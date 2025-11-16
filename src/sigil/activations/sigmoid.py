import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    The sigmoid function, can be used for many different tasks
    It maps any real-valued number into the (0, 1) interval.
    https://en.wikipedia.org/wiki/Sigmoid_function
    
    Parameters:
    - x: Input numpy array.
    
    Returns:
    - Numpy array after applying the sigmoid function element-wise.
    """
    return 1 / (1 + np.exp(-x))
