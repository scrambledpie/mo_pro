
import numpy as np

from .base import BaseOptimizer


class RandomWalker(BaseOptimizer):
    """
    Start with a random string, update one random location with a randomly
    selected amino acid.
    """
    def iterate(self):
        x, _ = self._xy_history[-1]
        idx = np.random.choice(self._input_length)
        aa = np.random.choice(self._amino_acids)
        x = x[:idx] + aa + x[idx+1:]
        y = self._fun(x)
        self._xy_history.append([x, y])
