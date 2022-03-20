import abc

import numpy as np
import matplotlib.pyplot as plt


class BaseOptimizer(abc.ABC):
    _amino_acids = list("RHKDESTNQCUGPAVILMFYW")
    def __init__(
        self, 
        fun: callable,
        input_length:int,
        num_iters: int,
        x_init = None
    ):
        self._fun = fun
        self._num_iters = num_iters
        self._input_length = input_length
        self._xy_history = []
        if x_init is not None:
            assert len(x_init) == input_length, "starting sequence is wrong length"
        else:
            x_init = "".join(np.random.choice(self._amino_acids, input_length))

        self.x_init = x_init

    @abc.abstractmethod
    def optimize(self):
        pass

    def get_pareto_points(self, y_matrix: np.ndarray=None):
        """ find the non-dominant points in the set of y vectors """
        if y_matrix is None:
            y_matrix = np.array([y for x, y in self._xy_history])

        y_pareto_a = y_pareto.T[:, None, :]  # (2, 1, n)
        y_pareto_b = y_pareto.T[:, :, None]  # (2, n, 1)
        dominance = np.all(y_pareto_a < y_pareto_b, axis=0)  # (2, n, n) -> (n, n)
        mask_dom = np.any(dominance, axis=1)  # (n, n) -> (n)
        indeces_dom = np.where(mask_dom==False)[0]  # (num_dominated)
        y_pareto = y_pareto[indeces_dom, :]  # noqa (num_dominated, 2)

        return y_pareto, indeces_dom

    @staticmethod
    def compute_2d_hypervolume(
        y_pareto: np.ndarray, y_max: np.ndarray, y_scale: np.ndarray, lower_is_better=True
    ) -> float:
        """
        Compute the hypervolume between the pareto front and the
        y_max point. First filter out the point that are dominated.
        With the remaining non? dominated points, draw a "staircase"
        and extend the last step and first step as far as y_max.
        From this extended staircase, close the volume back to
        include y_max. Compute the enclosed volume.

        Args:
            y (np.ndarray): 2D pareto front points
            anchor (np.ndarray): 2D point, the maximum y to include in hypervolume

        Returns:
            volume (float): the hypervolume
        """
        # clean inputs
        y_pareto = np.array(y_pareto).reshape(-1, 2)
        y_pareto = y_pareto / y_scale
        y_max = np.array(y_max).reshape(1, 2)
        y_max = y_max / y_scale

        if not lower_is_better:
            y_pareto = -y_pareto
            y_max = -y_max

        # filter out low value points
        mask_low = np.all(y_pareto <= y_max, axis=1)
        y_pareto = y_pareto[mask_low, :]

        # all points have been filtered..
        if y_pareto.shape[0] == 0:
            return 0.0

        # filter out dominated points
        y_pareto_a = y_pareto.T[:, None, :]  # (2, 1, n)
        y_pareto_b = y_pareto.T[:, :, None]  # (2, n, 1)
        dominance = np.all(y_pareto_a < y_pareto_b, axis=0)  # (2, n, n) -> (n, n)
        mask_dom = np.any(dominance, axis=1)  # (n, n) -> (n)
        y_pareto = y_pareto[mask_dom == False, :]  # noqa (num_dominated, 2)
        # import matplotlib.pyplot as plt
        # plt.plot(y_pareto[:, 0], y_pareto[:, 1], '^C1', alpha=0.5)

        # sort by first objective
        y_pareto = y_pareto[np.argsort(y_pareto[:, 0]), :]
        # scan from top left to bottom right, calculating the rectangle volume
        # from each dominant point.
        current_y1_max = y_max[0, 1]
        volume = 0
        for y in y_pareto:
            volume += np.abs(y[0] - y_max[0, 0]) * np.abs(y[1] - current_y1_max)
            current_y1_max = y[1]

        return volume

    @staticmethod
    def compute_average_linearisation(y_pareto: np.ndarray, y_scales: list = [1, 1]):
        """
        Take a collection of points in the 2d plane y_1,...,y_n and a
        collection of mixture weights lambda_1,....,lambda_k in [0, 1].
        For a given lambda_j value, we convert the set of points into single
        scores, for i = 0,....,n

            y_i_k = y_i[0] * lambda_j + y_i[1] * (1-lambda_j)

        we then take the point that has the best score which gives us the
        quality of the best point for this weight mixture lambda_k

            y_best_k = min_i y_i_k

        Finally, above is the score for a single weight mixture, we can compute
        this for a range of weight mixtures

            output = mean_k y_best_k
                = mean_k min_i y_i_k

        Args:
            y_pareto (np.ndarray): a (n, 2) matrix of points in the output space
            y_scales (list, optional): scaling factors for dimensions of y_pareto.
                Defaults to [1,1].
        """
        # sanity check inputs
        y_pareto = np.array(y_pareto).reshape(-1, 2)
        y_scales = np.array(y_scales).reshape(1, 2)

        # rescale y_values
        # (n, 2)
        y_pareto = y_pareto / y_scales

        # generate 11 different weight vectors
        # (2, 11)
        lambda_vecs = np.sqrt(
            np.vstack([np.linspace(0, 1, 11)[None, :], np.linspace(1, 0, 11)[None, :]])
        )
        # linearise two objectives into one with the different weight vectors
        # (n, 11)
        y_linearised = np.matmul(y_pareto, lambda_vecs)
        # compute the best linearised y for each lambda vector
        # (11)
        y_linearised_best = np.min(y_linearised, axis=0)
        # return the average of the linearised y values
        return np.mean(y_linearised_best)

    @staticmethod
    def compute_average_chebychev(
        y_pareto: np.ndarray, y_max: np.ndarray, y_scales: list = [1, 1]
    ):
        """
        Take a collection of points in the 2d plane y_1,...,y_n and a
        collection of mixture weights lambda_1,....,lambda_k in [0, 1].
        For a given lambda_j value, we convert the set of points into single
        scores, for all points i = 0,....,n

            y_i_k = max(y_i[0] * lambda_j, y_i[1] * (1-lambda_j))

        we then take the point that has the best score which gives us the
        quality of the best point for this weight mixture lambda_k

            y_best_k = min_i y_i_k

        Finally, above is the score for a single weight mixture, we can compute
        this for a range of weight mixtures

            output = mean_k y_best_k
                = mean_k min_i y_i_k

        Args:
            y_pareto (np.ndarray): [description]
            y_max: np.ndarray: [description]
            y_scales (list, optional): [description]. Defaults to [1,1].
        """

        # sanity check inputs
        y_pareto = np.array(y_pareto).reshape(-1, 2)
        y_max = np.array(y_max).reshape(1, 2)
        y_scales = np.array(y_scales).reshape(1, 2)

        # rescale and translate y_values
        # (n, 1, 2)
        y_pareto = (y_pareto - y_max) / y_scales
        y_pareto = y_pareto[:, None, :]  # (n, 1, 2)

        # generate 11 differnt weight vectors
        # (1, 11, 2)
        lambda_vecs = np.sqrt(
            np.hstack([np.linspace(0, 1, 11)[:, None], np.linspace(1, 0, 11)[:, None]])
        )
        lambda_vecs = lambda_vecs[None, :, :]

        # (n, 11, 2)
        y_lambda = y_pareto * lambda_vecs

        # (n, 11)
        y_scalarised = np.max(y_lambda, axis=2)

        # compute the best linearised y for each lambda vector
        # (11)
        y_linearised_best = np.min(y_scalarised, axis=0)

        # return the average of the linearised y values
        return np.mean(y_linearised_best)

    
    def plot_pareto(self, ax=None):
        y_matrix = np.vstack([y for x, y in self._xy_history])

        if not y_matrix.shape[1] == 2:
            return

        y_dominant, _ = self.get_pareto_points(y_matrix)
        print(y_dominant.shape)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.plot(y_matrix[:, 0], y_matrix[:, 1], label="y")
        ax.plot(y_dominant[:, 0], y_dominant[:, 1], label="pareto front")
        ax.legend()

    
    def plot_convergence(self):
        pass



        

        
