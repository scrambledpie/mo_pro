import sys
import os
import argparse

# an environment hack to ensure the other folders are in the python path
# TODO: upgrade to proper environment management: pipenv/conda/docker
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])


import matplotlib.pyplot as plt
import numpy as np

from multi_obj_funs.toy_funs import ToyFun
from multi_obj_funs.protein_function_predictors import ProteinFunctionPredictor

from optimizers.random_search import RandomSearch
from optimizers.random_walker import RandomWalker

# optimizers classes that may be used
OPTIMIZER_DICT = {
    "rs": RandomSearch,
    "rw" : RandomWalker,
}

# test functions (classes and constructor arguments) to be used.
FUNS_DICT = {
    "toy_10_2": {
        "cls": ToyFun,
        "kwargs":{"input_length":10, "num_objectives":2},
    },
    "toy_20_4": {
        "cls": ToyFun,
        "kwargs":{"input_length":20, "num_objectives":4},
    },
    "google_prot": {
        "cls": ProteinFunctionPredictor,
        "kwargs":{},
    }
}


if __name__=="__main__":
    # define command line args, then parse and sanity check them
    parser = argparse.ArgumentParser(description='run optimizer an obj function')
    parser.add_argument('--obj', type=str, help="objective", default="toy_10_2")
    parser.add_argument('--opt', type=str, help='optimiser', default="rw")
    parser.add_argument('--n', type=int, help='number of iterations', default=100)
    parser.add_argument('--o', type=str, help='output file', default=None)
    parser.add_argument('--seed', type=int, help='np rng seed', default=1)

    args = parser.parse_args()
    assert args.obj in FUNS_DICT, f"--obj must be in {list(FUNS_DICT.keys())}"
    assert args.opt in OPTIMIZER_DICT, f"--opt must be in {list(OPTIMIZER_DICT.keys())}"
    
    np.random.seed(args.seed)

    obj_fun = FUNS_DICT[args.obj]["cls"](**FUNS_DICT[args.obj]["kwargs"])
    optimiser = OPTIMIZER_DICT[args.opt](
        fun = obj_fun,
        input_length=obj_fun.input_length,
        budget=100,
    )

    optimiser.initialize_x()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    fig.suptitle(f"{optimiser.__class__.__name__} applied to {str(obj_fun)}")
    plt.ion()

    for _ in range(100):
        optimiser.iterate()
        optimiser.update_metrics()
        optimiser.plot_metrics(ax)
        fig.canvas.draw()
        fig.tight_layout()
        plt.show()
        plt.pause(0.5)

