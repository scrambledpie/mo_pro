import sys
import os
import argparse
import pickle as pkl
import datetime
import socket
import getpass
import time

from utils import check_write_permission


# an environment hack to ensure the other folders are in the python path
# TODO: upgrade to proper environment management: pipenv/conda/docker
sys.path.append(os.path.split(os.path.split(__file__)[0])[0])

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


def run_benchmark(
    obj_fun: callable = ToyFun(2, 5),
    optimiser_cls = RandomSearch,
    optimizer_kwargs: dict = {},
    budget: int = 100,
):
    """_summary_

    Parameters
    ----------
    obj_fun : callable
        an objective function, eg x="GRHYAGRHHHPGOPGRARG"; y = obj_fun(x)
    optimiser_cls
        the optimizer class, an instance of this class will be constructed and
        used to perform optimization.
    optimizer_args : dict, optional
        a dictionary of kwargs to be passed to the optimizer constructor
    budget : int
        the number of times to call obj_fun(x)
    meta_data : dict, optional
        any optional extra information to be saved to the output file, start/end
        time, local file location, computer being run etc
    """
    # keep the input args just for tracking
    benchmark_args = {k:str(v) for k, v in locals().items()}

    # instantiate the optimizer
    tick = time.time()
    optimizer = optimiser_cls(
        fun=obj_fun,
        input_length=obj_fun.input_length,
        budget = budget,
        **optimizer_kwargs
    )
    constructor_time = time.time() - tick

    # run the optimizer
    tick = time.time()
    optimizer.optimize()
    optimisation_time = time.time() - tick

    output = {
        "args": benchmark_args,
        "obj_fun":str(obj_fun),
        "opt_cls":str(optimiser_cls),
        "xy_history": optimizer._xy_history,
        "constructor_time": constructor_time,
        "optimization_time": optimisation_time,
    }
    
    return output


if __name__=="__main__":
    # define command line args, then parse and sanity check them
    parser = argparse.ArgumentParser(description='run optimizer an obj function')
    parser.add_argument('--obj', type=str, help="objective", default="toy_10_2")
    parser.add_argument('--opt', type=str, help='optimiser', default="rw")
    parser.add_argument('--n', type=int, help='number of iterations', default=100)
    parser.add_argument('--o', type=str, help='output file', default=None)
    args = parser.parse_args()
    assert args.obj in FUNS_DICT, f"--obj must be in {list(FUNS_DICT.keys())}"
    assert args.opt in OPTIMIZER_DICT, f"--opt must be in {list(OPTIMIZER_DICT.keys())}"
    if args.o is not None:
        assert check_write_permission(args.o), f"{args.o} is not writeable"

    # just keep a load of extra meta_data we may or may not need
    meta_data = {
        "cli_args": args,
        "__file__": __file__,
        "hostname":socket.gethostname(),
        "user": getpass.getuser(),
        "start_time": str(datetime.datetime.now())[:-7]
    }

    # lets get this party started!
    output = run_benchmark(
        obj_fun=FUNS_DICT[args.obj]["cls"](**FUNS_DICT[args.obj]["kwargs"]),
        optimiser_cls=OPTIMIZER_DICT[args.opt],
        optimizer_kwargs={},
        budget=args.n,
    )

    # save the output if a file is given
    meta_data["end_time"] = str(datetime.datetime.now())[:-7]
    output["meta_data"] = meta_data
    if args.o is not None:
        with open(args.o, "wb") as f:
            pkl.dump(output, f)
