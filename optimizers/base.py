import abc

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    get_pareto_points,
    compute_2d_hypervolume,
    compute_average_linearisation,
    compute_average_chebychev,
)


class BaseOptimizer(abc.ABC):
    """
    Common code for all optimizers goes here. This class serves two purposes
        1. the common interface by which all optimizers must abide, the same
          __init__() arguments and the self.optimize() that does the work.
        2. common optimiser utility functions: hypervolume, scalarisations,
          finding pareto front etc.
    """
    _amino_acids = list("RHKDESTNQCUGPAVILMFYW")

    def __init__(
        self, 
        fun: callable,
        input_length:int,
        budget: int
    ):
        """
        Parameters
        ----------
        fun : callable
            an objective function, eg x="MGRPQAHYHTRA",  y = fun(x)
            that returns a vector output y representing multiple objective
            values.
        input_length : int
            the length of the proteins to be passed into fun()
        budget : int
            number of optimization iterations to do
        """
        self._fun = fun
        self._budget = budget
        self._input_length = input_length
        self._xy_history = []
        self._metrics = {
            "N":[],
            "hypervolume":[],
            "linear_scalar":[],
            "cheby_scalar":[],
        }

    
    def optimize(self):
        """
        All optimizers must have an optimizer method that performs the search
        over protein string space.
        """
        self.initialize_x()

        while len(self._xy_history) < self._budget:
            n_before = len(self._xy_history)
            self.iterate()
            assert len(self._xy_history) > n_before, \
                "self.iterate() must add to self._xy_history!"
            self.update_metrics()


    def initialize_x(self, x_init:str=None):
        """ if not given, default initialization is a single random string """
        if x_init is not None:
            self.fun._validate_x(x_init)
        else:
            x_init = "".join(
                np.random.choice(
                    self._amino_acids,
                    self._input_length
                    )
                )
        y_init = self._fun(x_init)
        self._xy_history.append([x_init, y_init])


    @abc.abstractmethod
    def iterate(self):
        """
        All optimizers must implement an iterate function that uses the 
        _xy_history to determine new x values their y=f(x) values which are
        then added to the history. The intended code should look like

            # get last points
            x_last, y_last = self._xy_history[-1]

            # get new points
            x_new = ??? some_clever_computation using the _xy_history ????
            y_new = self._fun(x_new)

            # update history
            self._xy_history.append([x_new, y_new])
        """
        pass

    def update_metrics(self):
        """ compute convergence metrics and store the values """
        kwargs = dict(
            y_pareto = np.vstack([y for _, y in self._xy_history]),
            y_max = np.array([[4] * self._fun.num_objectives]),
            y_scales = np.array([[1] * self._fun.num_objectives]),
        )
        self._metrics["N"].append(len(self._xy_history))
        self._metrics["hypervolume"].append(compute_2d_hypervolume(**kwargs))
        self._metrics["linear_scalar"].append(compute_average_linearisation(**kwargs))
        self._metrics["cheby_scalar"].append(compute_average_chebychev(**kwargs))

    def plot_metrics(self, axes: list):
        """
        Provide a list of 4 matplotlib axes objects to show four convergence
        output plots.
        """
        if not len(self._xy_history[0][1]) == 2:
            # only 2d y vectors are supported
            return

        for ax in axes: ax.clear()

        y_matrix = np.vstack([y for _, y in self._xy_history])
        y_dominant, _ = get_pareto_points(y_matrix)
        axes[0].scatter(y_matrix[-1, 0], y_matrix[-1, 1], label="newest y", s=250)
        axes[0].scatter(y_matrix[:, 0], y_matrix[:, 1], label="y")
        axes[0].scatter(y_dominant[:, 0], y_dominant[:, 1], label="pareto front")
        axes[0].legend()
        axes[0].set_title("y points")
        axes[0].set_xlabel("y_1")
        axes[0].set_ylabel("y_2")

        for ax, metric in zip(
            axes[1:],
            ["hypervolume", "linear_scalar", "cheby_scalar"],
        ):
            ax.plot(self._metrics["N"], self._metrics[metric])
            ax.set_title(metric)
            ax.set_xlabel("Budget")

