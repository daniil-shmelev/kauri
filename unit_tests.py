import unittest
from trees import *
from maps import *

trees = [Tree(None),
         Tree([]),
         Tree([[]]),
         Tree([[],[]]),
         Tree([[[]]]),
         Tree([[],[],[]]),
         Tree([[],[[]]]),
         Tree([[[],[]]]),
         Tree([[[[]]]])]

class GeneralTests(unittest.TestCase):

    def test_conversion(self):
        for t in trees:
            self.assertEqual(repr(t), repr(t.asForest()), repr(t) + " " + repr(t.asForest()))
            self.assertEqual("1*" + repr(t), repr(t.asForestSum()), repr(t) + " " + repr(t.asForestSum()))

    def test_numNodes(self):
        nums = [0,1,2,3,3,4,4,4,4]
        for t, n in zip(trees, nums):
            self.assertEqual(n, t.numNodes(), repr(t) + " Tree")
            self.assertEqual(n, t.asForest().numNodes(), repr(t) + " Forest")

    def test_factorial(self):
        factorials = [1,1,2,3,6,4,8,12,24]
        for t, f in zip(trees, factorials):
            self.assertEqual(f, t.factorial(), repr(t) + " Tree")
            self.assertEqual(f, t.asForest().factorial(), repr(t) + " Forest")
            self.assertEqual(f, t.asForestSum().factorial(), repr(t) + " ForestSum")

    def test_antipode(self):
        antipodes = [
            1*Tree(None),
            -Tree([]),
            Tree([]) * Tree([]) - Tree([[]]),
            -Tree([]) * Tree([]) * Tree([]) + 2 * Tree([[]]) * Tree([]) - Tree([[],[]]),
            -Tree([]) * Tree([]) * Tree([]) + 2 * Tree([[]]) * Tree([]) - Tree([[[]]]),
            Tree([]) * Tree([]) * Tree([]) * Tree([]) - 3 * Tree([[]]) * Tree([]) * Tree([]) + 3 * Tree([[],[]]) * Tree([]) - Tree([[],[],[]]),
            Tree([]) * Tree([]) * Tree([]) * Tree([]) - 3 * Tree([[]]) * Tree([]) * Tree([]) + Tree([[],[]]) * Tree([]) + Tree([[]]) * Tree([[]]) + Tree([[[]]]) * Tree([]) - Tree([[],[[]]])
        ]

        for t, s in zip(trees[:7], antipodes):
            self.assertEqual(s, t.antipode(), repr(t) + " Tree")
            self.assertEqual(s, t.asForest().antipode(), repr(t) + " Forest")
            self.assertEqual(s, t.asForestSum().antipode(), repr(t) + " ForestSum")

    def test_antipode_property(self):
        for t in trees:
            self.assertEqual(counit(t), t.apply_product(S, ident))

    def test_id_sqrt(self):
        for t in trees:
            self.assertEqual(t, t.apply_product(sqrt, sqrt))

    def test_minus(self):
        for t in trees:
            self.assertAlmostEqual(t.apply(exact_weights), t.minus().apply(exact_weights))

    def test_plus(self):
        for t in trees:
            self.assertAlmostEqual(t.apply(counit), t.plus().apply(exact_weights))

    def test_plus_minus(self):
        for t in trees:
            self.assertEqual(t, t.apply_product(even_component, odd_component))


if __name__ == '__main__':
    unittest.main()