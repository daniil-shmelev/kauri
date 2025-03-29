import unittest
from rootedtrees import *

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

    def test_mixed_arithmetic(self):
        t0 = Tree(None)
        t1 = Tree([])
        t2 = Tree([[]])
        t3 = Tree([[], []])

        f1 = Forest([t1, t2])
        f2 = Forest([t3])

        self.assertTrue(f1 == (t1 * t2))

        s1 = ForestSum([f1, f2], [2, -1])  # 2*f1 - f2
        s2 = s1 + 5 * t0

        self.assertTrue(s1 == (2 * f1 - t3), repr(s1) + ", " + repr(2 * f1 - t3))
        self.assertTrue(s2 == (2 * f1 - t3 + 5), repr(s2) + ", " + repr(2 * f1 - t3 + 5))

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

    def test_antipode_squared(self):
        for t in trees:
            self.assertEqual(t, t.antipode().antipode())

    def test_antipode_squared_2(self):
        def f(t):
            n = t.numNodes()
            return t.apply_power(lambda x : x - x.antipode().antipode(), n)

        for t in trees[1:]:
            self.assertEqual(0, t.apply(f))

    def test_antipode_squared_3(self):
        def f(t):
            n = t.numNodes()
            f1 = lambda x : x + x.antipode()
            f2 = lambda x : x - x.antipode().antipode()
            f3 = lambda x : x.apply_power(f2, n-1).apply(f1)
            return t.apply(f3)

        for t in trees[1:]:
            self.assertEqual(0, t.apply(f))

    def test_adjoint_flow(self):
        for t in trees:
            self.assertAlmostEqual(t.apply(exact_weights), t.antipode().sign().apply(exact_weights))

    def test_RK_elementary_weights(self):
        #Test using an RK method of order 4
        A = [[0,0,0,0],
             [1./2,0,0,0],
             [0,1./2,0,0],
             [0,0,1,0]]
        b = [1./6,1./3,1./3,1./6]

        f = lambda x : RK_elementary_weights(x, A, b)

        for t in trees:
            self.assertAlmostEqual(t.apply(exact_weights), t.apply(f))


if __name__ == '__main__':
    unittest.main()