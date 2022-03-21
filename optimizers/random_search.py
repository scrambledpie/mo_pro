import numpy as np

from .base import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    Generate a new random string in each iteration. Good for cheaply testing
    workflows and as a naice control baseline.
    """
    def iterate(self): 
        x = "".join(np.random.choice(self._amino_acids, self._input_length))
        y = self._fun(x)
        self._xy_history.append([x, y])