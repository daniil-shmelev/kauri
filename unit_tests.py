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

class TreeTests(unittest.TestCase):

    def test_conversion(self):
        for t in trees:
            self.assertEqual(repr(t), repr(t.as_forest()), repr(t) + " " + repr(t.as_forest()))
            self.assertEqual("1*" + repr(t), repr(t.as_forest_sum()), repr(t) + " " + repr(t.as_forest_sum()))

    def test_add(self):
        t1 = Tree([]) + Tree([[],[]])
        t2 = ForestSum([
            Forest([Tree([])]),
            Forest([Tree([[],[]])])
        ], [1,1])
        self.assertEqual(t2, t1)

        t1 = Tree([]) - 2 * Tree([[], []])
        t2 = ForestSum([
            Forest([Tree([])]),
            Forest([Tree([[], []])])
        ], [1, -2])
        self.assertEqual(t2, t1)

        t1 = 1 + Tree([[], []])
        t2 = ForestSum([
            Forest([Tree(None)]),
            Forest([Tree([[], []])])
        ], [1, 1])
        self.assertEqual(t2, t1)

        t1 = Tree([[], []]) + 2
        t2 = ForestSum([
            Forest([Tree([[], []])]),
            Forest([Tree(None)])
        ], [1, 2])
        self.assertEqual(t2, t1)

        t1 = Tree([[], []]) + Forest([Tree([]), Tree([[]])])
        t2 = ForestSum([
            Forest([Tree([[], []])]),
            Forest([Tree([]), Tree([[]])])
        ], [1, 1])
        self.assertEqual(t2, t1)

    def test_mul(self):
        t1 = Tree([]) * Tree([[], []])
        t2 = Forest([Tree([]), Tree([[],[]])])
        self.assertEqual(t2, t1)

        t1 = Tree([]) * Tree(None) * Tree(None)
        t2 = Tree([])
        self.assertEqual(t1, t2)

        t1 = Tree([[]]) * Forest([Tree([]), Tree([[],[]])])
        t2 = Forest([Tree([[]]), Tree([]), Tree([[],[]])])
        self.assertEqual(t1, t2)

        t1 = (Tree([]) - Tree([[]])) * Tree([])
        t2 = Tree([]) * Tree([]) - Tree([[]]) * Tree([])
        self.assertEqual(t1,t2)

    def test_pow_tree(self):
        t1 = Tree([]) ** 3
        t2 = Tree([]) * Tree([]) * Tree([])
        self.assertEqual(t1, t2)

        t1 = Tree(None) ** 3
        self.assertEqual(1, t1)

        t1 = Tree([]) ** 0
        self.assertEqual(1, t1)

    def test_pow_forest(self):
        t1 = Forest([Tree([]), Tree(None)]) ** 3
        t2 = Tree([]) * Tree([]) * Tree([])
        self.assertEqual(t1, t2)

        t1 = Tree(None).as_forest() ** 3
        self.assertEqual(1, t1)

        t1 = Tree([]).as_forest() ** 0
        self.assertEqual(1, t1)

    def test_pow_forest_sum(self):
        t1 = (Tree([]) - 2 * Tree([[]])) ** 2
        t2 = Tree([]) * Tree([]) - 4 * Tree([]) * Tree([[]]) + 4 * Tree([[]]) * Tree([[]])
        self.assertEqual(t1, t2)

        t1 = Tree([]).as_forest_sum() ** 0
        self.assertEqual(1, t1)

    def test_equality(self):
        self.assertEqual(Tree([[],[[]]]), Tree([[[]],[]]))
        self.assertEqual(Tree([[], [[]]]).as_forest(), Tree([[[]], []]))
        self.assertEqual(Tree([[],[[],[]]]), Tree([[[],[]], []]))
        self.assertEqual(Tree([[[]],[],[]]), Tree([[],[[]],[]]))
        self.assertEqual(Tree([[[]], [], []]), Tree([[], [], [[]]]))
        self.assertEqual(Forest([Tree([[]]), Tree([]), Tree([[],[]])]), Forest([Tree([]), Tree([[],[]]), Tree([[]])]))
        self.assertEqual(Tree([]) - Tree([[]]), -Tree([[]]) + Tree([]))
        self.assertEqual(ForestSum([Tree([[]]).as_forest(), Tree([]).as_forest(), Tree([]).as_forest()],[1,1,-1]), Tree([[]]).as_forest_sum())

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

    def test_num_nodes(self):
        nums = [0,1,2,3,3,4,4,4,4]
        for t, n in zip(trees, nums):
            self.assertEqual(n, t.nodes(), repr(t) + " Tree")
            self.assertEqual(n, t.as_forest().nodes(), repr(t) + " Forest")

    def test_sigma(self):
        vals = [1, 1, 1, 1, 2, 1, 2, 1, 6, 1, 2, 1, 1, 6, 2, 2, 2, 24]
        i = 0
        for t in trees_up_to_order(5):
            self.assertEqual(vals[i], t.sigma(), "i = " + str(i))
            i += 1

    def test_alpha(self):
        vals = [1, 1, 2, 6, 24, 120, 720]
        for n in range(1, 8):
            alpha_sum = 0
            for t in trees_of_order(n):
                alpha_sum += t.alpha()
            self.assertEqual(vals[n-1], alpha_sum)

    def test_beta(self):
        vals = [1,2,9,64,625,7776,117649]
        for n in range(1, 8):
            beta_sum = 0
            for t in trees_of_order(n):
                beta_sum += t.beta()
            self.assertEqual(vals[n-1], beta_sum)

    def test_factorial(self):
        factorials = [1,1,2,3,6,4,8,12,24]
        for t, f in zip(trees, factorials):
            self.assertEqual(f, t.factorial(), repr(t) + " Tree")
            self.assertEqual(f, t.as_forest().factorial(), repr(t) + " Forest")
            self.assertEqual(f, t.as_forest_sum().factorial(), repr(t) + " ForestSum")

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
            self.assertEqual(s, t.as_forest().antipode(), repr(t) + " Forest")
            self.assertEqual(s, t.as_forest_sum().antipode(), repr(t) + " ForestSum")

    def test_antipode_property(self):
        for t in trees:
            self.assertEqual(counit(t), t.apply_product(S, ident))

    def test_antipode_squared(self):
        for t in trees:
            self.assertEqual(t, t.antipode().antipode())

    def test_antipode_squared_2(self):
        def f(t):
            n = t.nodes()
            return t.apply_power(lambda x : x - x.antipode().antipode(), n)

        for t in trees[1:]:
            self.assertEqual(0, t.apply(f))

    def test_antipode_squared_3(self):
        def f(t):
            n = t.nodes()
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

        scheme = RK(A,b)

        for t in trees:
            self.assertAlmostEqual(t.apply(exact_weights), scheme.elementary_weights(t))

    def test_apply_power(self):
        pass


if __name__ == '__main__':
    unittest.main()