import numpy as np

from base import BaseOptimizer


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


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.append("/home/michael/mo_pro")
    from functions.toy_funs import ToyFun

    optimizer = RandomSearch(
        fun=ToyFun(),
        input_length=1000,
        num_iters=100,
    )
    optimizer.optimize()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    optimizer.plot_pareto(ax)

    plt.show()