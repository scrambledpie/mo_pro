import numpy as np

from .base import BaseFun


class ProteinFunctionPredictor(BaseFun):
    """
    TODO: the Google Research protein function predictor model
    """
    def __init__(self):
        # TODO: load the NN model
        self._nn_model = None

    def __call__(self, x: str) -> np.ndarray:
        # TODO
        pass

    @property
    def num_objectives(self):
        # TODO
        pass

    @property
    def input_length(self):
        # TODO
        pass
