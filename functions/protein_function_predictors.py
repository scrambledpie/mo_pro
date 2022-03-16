import numpy as np

from .base import BaseFun


class ProteinFunctionPredictor(BaseFun):
    """
    TODO: google protein funciton predictor model
    """
    def __init__(self):
        super().__init__()
        self._nn_model = None
    
    def __call__(self, x: str) -> np.ndarray:
        return self._nn_model(x)


