from .base import BaseFun
import pickle as pkl
import numpy as np


class ToyFun(BaseFun):

    def __init__(self, num_objectives=2):
        with open("toy_fun_weights.pkl", "rb") as f:
            self._weights = pkl.load(f)
        self._weights = self._weights[:num_objectives, :, :]
        
        int_list = np.arange(len(self._amino_acids))
        
        one_hot_dict = {}
        for i, aa in enumerate(self._amino_acids):
            aa_one_hot = 1 * (i==int_list)
            one_hot_dict[aa] = aa_one_hot
        self._one_hot_dict = one_hot_dict


    def __call__(self, x: str) -> list:
        self._validate_x(x)
        x_one_hot = np.vstack([self._one_hot_dict[aa] for aa in x])
        x_one_hot = x_one_hot[None, :, :]
        x_weights = self._weights[:, :x_one_hot.shape[1], :x_one_hot.shape[2]]
        return np.sum(x_one_hot * x_weights, axis=(1, 2))