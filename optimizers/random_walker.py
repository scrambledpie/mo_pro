
import numpy as np

from .base import BaseOptimizer


class RandomWalker(BaseOptimizer):
    """
    Start with a random string, randomly update one location
    with a randomly selected amino acid.
    This does not 
    """
    def optimize(self, x_init = None):
        
        if x_init is None:
            x_init = self.x_init

        x = x_init
        y = self._fun(x)
        self._xy_history.append([x, y])

        for _ in range(self._num_iters - 1):
            idx = np.random.choice(self._input_length)
            aa = np.random.choice(self._amino_acids)
            x = x[:idx] + aa + x[idx+1:]
            y = self._fun(x)
            self._xy_history.append([x, y])


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.append("/home/michael/mo_pro")
    from functions.toy_funs import ToyFun

    optimizer = RandomWalker(
        fun=ToyFun(),
        input_length=1000,
        num_iters=100,
    )
    optimizer.optimize()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    optimizer.plot_pareto(ax)

    plt.show()