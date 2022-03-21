from cmath import isnan
from termios import TIOCPKT_FLUSHREAD
import unittest
import sys
from pathlib import Path

# hack if the environment is not setup
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import numpy as np

from multi_obj_funs.toy_funs import ToyFun
from multi_obj_funs.protein_function_predictors import ProteinFunctionPredictor


class TestToyFun(unittest.TestCase):

    def fun_checker(self, fun: callable):
        """
        Take any objective function, call it a few times and do some basic
        sanity checks.
        """
        input_length = fun.input_length
        num_objectives = fun.num_objectives

        # execute fun with some randomly generated strings, sanity check output
        for _ in range(5):
            x = np.random.choice(list("RHKDESTNQCUGPAVILMFYW"), input_length)
            x = "".join(x)
            y = fun(x)
            self.assertFalse(any(np.isneginf(y)))
            self.assertFalse(any(np.isposinf(y)))
            self.assertFalse(any([isnan(y_i) for y_i in y]))
            self.assertEqual(len(y), num_objectives)
        
        # make sure invalid inputs raise errors
        with self.assertRaises(AssertionError):
            x = np.random.choice(list("RHKDESTNQCUGPAVILMFYW"), input_length + 1)
            x = "".join(x)
            y = fun(x)

        # make sure invalid inputs raise errors
        with self.assertRaises(AssertionError):
            x = np.random.choice(list("ZZZZZ"), input_length)
            x = "".join(x)
            y = fun(x)
        
        # make sure invalid inputs raise errors
        with self.assertRaises(AssertionError):
            y = fun(["house", 2, unittest])

    def test_toyfun_10_4(self):
        """ Run the toy function with 10 inputs and 4 outputs """
        fun = ToyFun(
            input_length=10,
            num_objectives=4,
        )
        self.fun_checker(fun)
    
    def test_toyfun_40_2(self):
        """ Run the toy function with 40 inputs and 2 outputs """
        fun = ToyFun(
            input_length=40,
            num_objectives=2,
        )
        self.fun_checker(fun)
    
    def test_protein_function_predictor(self):
        """ Check the protein_function_function_predictor """
        fun = ProteinFunctionPredictor()
        # TODO: implement the protein function predictor!
        # self.fun_checker(fun)


if __name__=="__main__":
    unittest.main()