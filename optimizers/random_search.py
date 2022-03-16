import numpy as np

from .base import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    Generate a new random string in each iteration. Good for cheaply testing
    workflows and as a naive baseline.
    """
    def optimize(self): 

        for _ in range(self._num_iters):
            x = "".join(np.random.choice(self._amino_acids, self._input_length))
            y = self._fun(x)
            self._xy_history.append([x, y])
        



        
