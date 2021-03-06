import numpy as np


def get_pareto_points(y_matrix: np.ndarray):
    """ find the non-dominant points in the set of y vectors """
    y_pareto_a = y_matrix.T[:, None, :]  # (num_objs, 1, n)
    y_pareto_b = y_matrix.T[:, :, None]  # (num_objs, n, 1)
    dominance = np.all(y_pareto_a < y_pareto_b, axis=0)  # (num_objs, n, n) -> (n, n)
    mask_dom = np.any(dominance, axis=1)  # (n, n) -> (n)
    indeces_dom = np.where(mask_dom==False)[0]  # (num_dominated)
    y_matrix = y_matrix[indeces_dom, :]  # noqa (num_dominated, num_objs)
    return y_matrix, indeces_dom


def compute_2d_hypervolume(
    y_pareto: np.ndarray,
    y_max: np.ndarray=np.array([[3, 3]]),
    y_scales: np.ndarray=np.array([[1, 1]]),
    lower_is_better=True
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
    assert y_pareto.shape[1] == 2, "only 2"
    # clean inputs
    y_pareto = np.array(y_pareto).reshape(-1, 2)
    y_pareto = y_pareto / y_scales
    y_max = np.array(y_max).reshape(1, 2)
    y_max = y_max / y_scales

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
    y_pareto, _ = get_pareto_points(y_pareto)

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


def compute_average_linearisation(
    y_pareto: np.ndarray,
    y_scales: list = [1, 1],
    y_max=None
):
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
        y_max (list) unused argument, just here for consistency with other methods
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


def compute_average_chebychev(
    y_pareto: np.ndarray,
    y_max: np.ndarray=[3, 3],
    y_scales: list = [1, 1]
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