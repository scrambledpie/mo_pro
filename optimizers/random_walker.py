import numpy as np

from .base import BaseOptimizer


class RandomWalker(BaseOptimizer):
    """
    Start with a random string, randomly update one location
    with a randomly selected amino acid.
    This does not 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize(self, x_init = None):
        
        if x_init is not None:
            x_init = self.x_init

        x = x_init
        y = self._fun(x)
        self._xy_history.append([x, y])

        for _ in range(self._num_iters):
            idx = np.random.choice(self._input_length)
            aa = np.random.choice(self._amino_acids)
            x = x[:idx] + aa + x[idx+1:]
            y = self._fun(x)
            self._xy_history.append([x, y])