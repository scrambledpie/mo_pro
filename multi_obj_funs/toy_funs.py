from .base import BaseFun
import pickle as pkl
import numpy as np
import os


WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "toy_fun_weights.pkl")


class ToyFun(BaseFun):
    """
    A function the takes protein strings as input and returns a quality score.
    The (very basic) computation is as follows
        - Take a protein string,
        - convert each amino acid to a one hot vector
        - stack vectors into a binary matrix
        - element-wise multiply by pregenerated random matrices
        - for each matrix sum elements to yield a score for one objective.
    """

    def __init__(self, num_objectives:int=2, input_length:int = 10):
        with open(WEIGHTS_FILE, "rb") as f:
            weights = pkl.load(f)
        assert num_objectives <= weights.shape[0], \
            f"num_objectives {num_objectives} > {weights.shape[0]}"
        assert input_length <= weights.shape[1], \
            f"input_length {input_length} > {weights.shape[1]}"

        # truncate weight matrix to (num_objectives, input_length, num_aa)
        self._weights = weights[
            :num_objectives,
            :input_length,
            :len(self._amino_acids)
        ]
        self._input_length = input_length
        self._num_objectives = num_objectives
        
        # make a dictionary from amino acid character to one hot vector
        # {"A": [1, 0, 0,....], "R": [0, 1,......],....}
        int_list = np.arange(len(self._amino_acids))
        aa_one_hot = {}
        for i, aa in enumerate(self._amino_acids):
            one_hot = 1 * (i==int_list)
            aa_one_hot[aa] = one_hot
        self._one_hot_dict = aa_one_hot

    @property
    def input_length(self):
        return self._input_length

    @property
    def num_objectives(self):
        return self._num_objectives
    
    def __repr__(self):
        return f"ToyFun_{self.input_length}_{self._num_objectives}"
    
    def __str__(self):
        return self.__repr__()

    def __call__(self, x: str) -> list:
        self._validate_x(x)

        # make a matrix of stacked one-hot vectors (len(x), 20)
        x_one_hot = np.vstack([self._one_hot_dict[aa] for aa in x])

        # (len(x), 20) -> (1, len(x), 20)
        x_one_hot = x_one_hot[None, :, :]

        # (1, len(x), 20) * (num_obj, len(x), 20) -> sum -> (num_obj)
        output = np.mean(x_one_hot * self._weights, axis=(1, 2))

        return output