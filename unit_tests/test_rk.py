import unittest
from kauri import *
from kauri import Tree as T

trees = [T(None),
         T([]),
         T([[]]),
         T([[],[]]),
         T([[[]]]),
         T([[],[],[]]),
         T([[],[[]]]),
         T([[[],[]]]),
         T([[[[]]]])]

class RKTests(unittest.TestCase):
    def test_RK_elementary_weights(self):
        # Test using an RK method of order 4
        A = [[0, 0, 0, 0],
             [1. / 2, 0, 0, 0],
             [0, 1. / 2, 0, 0],
             [0, 0, 1, 0]]
        b = [1. / 6, 1. / 3, 1. / 3, 1. / 6]

        scheme = RK(A, b)
        rk_weights = scheme.elementary_weights_map()

        for t in trees:
            self.assertAlmostEqual(exact_weights(t), rk_weights(t))