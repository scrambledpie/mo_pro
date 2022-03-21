import sys
import os
import argparse
import pickle as pkl

# an environment hack to ensure the other folders are in the python path
# TODO: upgrade to proper environment management: pipenv/conda/docker
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])

from functions.toy_funs import ToyFun
from functions.protein_function_predictors import ProteinFunctionPredictor

from optimizers.random_search import RandomSearch
from optimizers.random_walker import RandomWalker

def run_benchmark(
    optimiser_cls,
    obj_fun_cls,
    budget: int,
    output_file: str,
    obj_fun_args: dict={},
    optimizer_args: dict={},
    meta_data:dict={},
):
    """_summary_

    Parameters
    ----------
    optimiser_cls : _type_
        _description_
    obj_fun_cls : _type_
        _description_
    budget : int
        _description_
    output_file : str
        _description_
    obj_fun_args : dict, optional
        _description_, by default {}
    optimizer_args : dict, optional
        _description_, by default {}
    meta_data : dict, optional
        _description_, by default {}
    """
    args = {k:str(v) for k, v in locals()}

    obj_fun = obj_fun_cls(**obj_fun_args)

    optimizer = optimiser_cls(
        fun=obj_fun,
        input_length=obj_fun.input_length,
        num_iters = budget,
        **optimizer_args
    )

    optimizer.optimize()

    output = {
        "args": args,
        "xy_history": optimizer._xy_history,
        "meta_data":meta_data,
    }
    with open(output_file, "wb") as f:
        pkl.dump(output, f)
    



if __name__=="__main__":
    pass

