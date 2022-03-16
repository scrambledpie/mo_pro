import unittest
import sys

sys.path.append("../")  # just a hack

from optimizers.random_walker import RandomWalker
from optimizers.random_search import RandomSearch

from functions.toy_funs import ToyFun


class TestOptimizers(unittest.TestCase):
    def setUp(self) -> None:
        self._fun = ToyFun()

    def test_random_search(self):
        pass

    def validate_history(history:list, input_length:int):
        pass



