import unittest
import sys
from pathlib import Path

from cv2 import matMulDeriv

# hack if the environment is not setup
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from optimizers.random_walker import RandomWalker
from optimizers.random_search import RandomSearch

from multi_obj_funs.toy_funs import ToyFun


class TestOptimizers(unittest.TestCase):

    def run_optimizer(self, opt_cls):
        optimizer = opt_cls(
            fun=ToyFun(input_length=100),
            input_length=100,
            budget=100,
        )
        optimizer.optimize()
        xy = optimizer._xy_history
        self.assertTrue(len(xy)>=100)
        self.assertTrue(all([len(x)==100 for x, y in xy]))

    def test_random_search(self):
        self.run_optimizer(RandomSearch)

    def test_random_walker(self):
        self.run_optimizer(RandomWalker)
    
    def test_my_badass_optimiser(self):
        # self.run_optimizer(MyBadassOptimzerClass)
        pass


if __name__=="__main__":
    unittest.main()

