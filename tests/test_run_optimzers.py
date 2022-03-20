import unittest
import sys
from pathlib import Path

# hack if the environment is not setup
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from optimizers.random_walker import RandomWalker
from optimizers.random_search import RandomSearch

from functions.toy_funs import ToyFun


class TestOptimizers(unittest.TestCase):

    def run_optimizer(self, opt_cls):
        optimizer = opt_cls(
            fun=ToyFun(),
            input_length=100,
            num_iters=100,
        )

        optimizer.optimize()

        xy = optimizer._xy_history

        self.assertEqual(len(xy), 100)
        self.assertTrue(all([len(x)==100 for x, y in xy]))
        optimizer.plot_pareto()

    def test_random_search(self):
        self.run_optimizer(RandomSearch)

    def test_random_walker(self):
        self.run_optimizer(RandomWalker)


if __name__=="__main__":
    tt = TestOptimizers()
    tt.test_random_search()
    tt.test_random_walker()