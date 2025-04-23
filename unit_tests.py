import unittest
from kauri import *

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

    def test_repr(self):
        self.assertEqual(repr(Tree([[[]], []])), '[[[]], []]')
        self.assertEqual(repr(Tree([[[]], []]).as_forest()), '[[[]], []]')
        self.assertEqual(repr(Tree([[[]], []]).as_forest_sum()), '1*[[[]], []]')
        self.assertEqual(repr(Tree(None)), "∅")
        self.assertEqual(repr(Forest([])), "∅")
        self.assertEqual(repr(ForestSum([])), "0")

    def test_conversion(self):
        for t in trees:
            self.assertEqual(repr(t), repr(t.as_forest()), repr(t) + " " + repr(t.as_forest()))
            self.assertEqual("1*" + repr(t), repr(t.as_forest_sum()), repr(t) + " " + repr(t.as_forest_sum()))

    def test_add(self):
        t1 = Tree([]) + Tree([[],[]])
        t2 = ForestSum((
            (1, Forest([Tree([])])),
            (1, Forest([Tree([[],[]])]))
        ))
        self.assertEqual(t2, t1)

        t1 = Tree([]) - 2 * Tree([[], []])
        t2 = ForestSum((
            (1, Forest([Tree([])])),
            (-2, Forest([Tree([[], []])]))
        ))
        self.assertEqual(t2, t1)

        t1 = 1 + Tree([[], []])
        t2 = ForestSum((
            (1, Forest([Tree(None)])),
            (1, Forest([Tree([[], []])]))
        ))
        self.assertEqual(t2, t1)

        t1 = Tree([[], []]) + 2
        t2 = ForestSum((
            (1, Forest([Tree([[], []])])),
            (2, Forest([Tree(None)]))
        ))
        self.assertEqual(t2, t1)

        t1 = Tree([[], []]) + Forest([Tree([]), Tree([[]])])
        t2 = ForestSum((
            (1, Forest([Tree([[], []])])),
            (1, Forest([Tree([]), Tree([[]])]))
        ))
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

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(ValueError):
            t1 ** 1.5

    def test_pow_forest(self):
        t1 = Forest([Tree([]), Tree(None)]) ** 3
        t2 = Tree([]) * Tree([]) * Tree([])
        self.assertEqual(t1, t2)

        t1 = Tree(None).as_forest() ** 3
        self.assertEqual(1, t1)

        t1 = Tree([]).as_forest() ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(ValueError):
            t1 ** 1.5

    def test_pow_forest_sum(self):
        t1 = (Tree([]) - 2 * Tree([[]])) ** 2
        t2 = Tree([]) * Tree([]) - 4 * Tree([]) * Tree([[]]) + 4 * Tree([[]]) * Tree([[]])
        self.assertEqual(t1, t2)

        t1 = Tree([]).as_forest_sum() ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(ValueError):
            t1 ** 1.5

    def test_equality(self):
        self.assertEqual(Tree([[],[[]]]), Tree([[[]],[]]))
        self.assertEqual(Tree([[], [[]]]).as_forest(), Tree([[[]], []]))
        self.assertEqual(Tree([[],[[],[]]]), Tree([[[],[]], []]))
        self.assertEqual(Tree([[[]],[],[]]), Tree([[],[[]],[]]))
        self.assertEqual(Tree([[[]], [], []]), Tree([[], [], [[]]]))
        self.assertEqual(Forest([Tree([[]]), Tree([]), Tree([[],[]])]), Forest([Tree([]), Tree([[],[]]), Tree([[]])]))
        self.assertEqual(Tree([]) - Tree([[]]), -Tree([[]]) + Tree([]))
        self.assertEqual(ForestSum([(1, Tree([[]]).as_forest()), (1, Tree([]).as_forest()), (-1, Tree([]).as_forest())]), Tree([[]]).as_forest_sum())

    def test_equality_2(self):
        self.assertEqual(
            2*Tree([[]]) * Tree([]),
            2*Tree([]) * Tree([[]])
        )

        self.assertEqual(
            -Tree([]) * Tree([]) * Tree([]) + 2 * Tree([[]]) * Tree([]) - Tree([[], []]),
            - Tree([[], []]) + 2 * Tree([]) * Tree([[]]) -Tree([]) * Tree([]) * Tree([])
        )

    def test_hash(self):
        self.assertEqual(hash(Tree([[],[[]]])), hash(Tree([[[]],[]])))
        self.assertEqual(hash(Tree([[],[[],[]]])), hash(Tree([[[],[]], []])))
        self.assertEqual(hash(Tree([[[]],[],[]])), hash(Tree([[],[[]],[]])))
        self.assertEqual(hash(Tree([[[]], [], []])), hash(Tree([[], [], [[]]])))
        self.assertEqual(hash(Forest([Tree([[]]), Tree([]), Tree([[],[]])])), hash(Forest([Tree([]), Tree([[],[]]), Tree([[]])])))
        self.assertEqual(hash(Tree([]) - Tree([[]])), hash(-Tree([[]]) + Tree([])))
        self.assertEqual(hash(Tree([[]]) + Tree([]) - Tree([])), hash(Tree([[]]).as_forest_sum()))

    def test_mixed_arithmetic(self):
        t0 = Tree(None)
        t1 = Tree([])
        t2 = Tree([[]])
        t3 = Tree([[], []])

        f1 = Forest([t1, t2])
        f2 = Forest([t3])

        self.assertTrue(f1 == (t1 * t2))

        s1 = ForestSum((  (2,f1), (-1, f2)  ))  # 2*f1 - f2
        s2 = s1 + 5 * t0

        self.assertTrue(s1 == (2 * f1 - t3), repr(s1) + ", " + repr(2 * f1 - t3))
        self.assertTrue(s2 == (2 * f1 - t3 + 5), repr(s2) + ", " + repr(2 * f1 - t3 + 5))

    def test_nodes(self):
        nums = [0,1,2,3,3,4,4,4,4]
        for t, n in zip(trees, nums):
            self.assertEqual(n, t.nodes(), repr(t) + " Tree")
            self.assertEqual(n, t.as_forest().nodes(), repr(t) + " Forest")
            self.assertEqual(n, t.as_forest_sum().nodes(), repr(t) + " Forest")

    def test_height(self):
        nums = [0,1,2,2,3,2,3,3,4]
        for t, n in zip(trees, nums):
            self.assertEqual(n, t.height(), repr(t))

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
            self.assertEqual(s, bck.antipode(t), repr(t) + " Tree")
            self.assertEqual(s, bck.antipode(t.as_forest()), repr(t) + " Forest")
            self.assertEqual(s, bck.antipode(t.as_forest_sum()), repr(t) + " ForestSum")

    def test_antipode_property(self):
        m = bck.map_product(bck.antipode, ident)
        for t in trees:
            self.assertEqual(bck.counit(t), m(t))

    def test_antipode_squared(self):
        f = bck.antipode
        g = f @ f
        for t in trees:
            self.assertEqual(t, g(t))

    def test_antipode_squared_2(self):
        f = bck.antipode
        g = f @ f

        for t in trees[1:]:
            self.assertEqual(0, ((ident - g) ** t.nodes())(t))

    def test_antipode_squared_3(self):
        f = bck.antipode
        g = f @ f

        h = Map(lambda x: ((ident - g) ** (x.nodes() - 1))(x))
        m = (ident + f) @ h

        for t in trees[1:]:
            self.assertEqual(0, m(t))

    def test_exact_weights(self):
        m1 = exact_weights ** 2
        m2 = Map(lambda x : m1(x) / 2**x.nodes())
        m3 = exact_weights ** (-1)
        m4 = Map(lambda x : m3(x) * (-1) ** x.nodes())
        for t in trees:
            self.assertAlmostEqual(exact_weights(t), m2(t))
            self.assertAlmostEqual(exact_weights(t), m4(t))

    def test_adjoint_flow(self):
        for t in trees:
            self.assertAlmostEqual(exact_weights(t), exact_weights(bck.antipode(t).sign()))

    def test_RK_elementary_weights(self):
        #Test using an RK method of order 4
        A = [[0,0,0,0],
             [1./2,0,0,0],
             [0,1./2,0,0],
             [0,0,1,0]]
        b = [1./6,1./3,1./3,1./6]

        scheme = RK(A,b)

        for t in trees:
            self.assertAlmostEqual(exact_weights(t), scheme.elementary_weights(t))

    def test_apply_power(self):
        S = bck.antipode
        m1 = ((S * S) * S)
        m2 = S ** 3
        for t in trees:
            self.assertEqual(m1(t), m2(t))

    def test_apply_negative_power(self):
        func_ = Map(lambda x : x**2)
        func3_ = func_ ** 3
        func_neg_3_ = func_ ** (-3)
        m = func3_ * func_neg_3_
        for t in trees:
            self.assertEqual(bck.counit(t), m(t))

    def test_apply_negative_power_scalar(self):
        func_ = Map(lambda x : x.nodes() if x.list_repr is not None else 1)
        func3_ = func_ ** 3
        func_neg_3_ = func_ ** (-3)
        m = func3_ * func_neg_3_
        for t in trees:
            self.assertEqual(bck.counit(t), m(t))

class MapTests(unittest.TestCase):
    def test_mul(self):
        f = Map(lambda x : x)
        g = bck.antipode
        h = f * g

        for t in trees:
            self.assertEqual(kauri.bck.bck.counit(t), h(t))

    def test_add(self):
        f = bck.antipode
        g = -f

        for t in trees:
            self.assertEqual(0, (g+f)(t))

    def test_inverse(self):
        f = Map(lambda x : x.nodes() if x.list_repr is not None else 1)

        g = f * (f**(-1))
        for t in trees:
            self.assertEqual(kauri.bck.bck.counit(t), g(t))

    def test_exact_weights(self):
        for t in trees:
            self.assertAlmostEqual(exact_weights(t), (exact_weights ** 2)(t) / 2**t.nodes())
            self.assertAlmostEqual(exact_weights(t), (exact_weights ** (-1))(t) * (-1) ** t.nodes())

    def test_composition(self):
        S = bck.antipode
        f = Map(lambda x : x**2)

        for t in trees:
            self.assertEqual((f @ S)(t), f(S(t)))

    def test_compose_scalar(self):
        f = lambda x : x.factorial()

        for t in trees:
            self.assertEqual(f(t), (kauri.bck.bck.counit @ f)(t))

    def test_inverse_identity(self):
        a = Map(lambda x : 1 if x == Tree(None) or x == Tree([]) else 0) ** (-1)
        for t in trees:
            self.assertEqual((-1)**t.nodes(), a(t), repr(t))

class CEMTests(unittest.TestCase):

    def test_coproduct(self):
        #TODO
        pass

    def test_antipode(self):
        trees_ = [
            Tree([]),
            Tree([[]]),
            Tree([[],[]]),
            Tree([[[]]])
        ]
        antipodes_ = [
            Tree([]),
            -Tree([[]]),
            -Tree([[],[]]) + 2 * Tree([[]])**2,
            -Tree([[[]]]) + 2 * Tree([[]])**2
        ]
        for t, a in zip(trees_, antipodes_):
            self.assertEqual(a, cem.antipode(t))

    def test_antipode_property(self):
        m = cem.antipode ^ ident
        for t in trees[1:]:
            self.assertEqual((cem.counit(t) * Tree([])), m(t), repr(t))

    def test_antipode_squared(self):
        f = cem.antipode
        g = f @ f
        for t in trees[1:]:
            self.assertEqual(t, g(t))

    def test_antipode_squared_2(self):
        f = cem.antipode
        g = f @ f

        for t in trees[1:]:
            self.assertEqual(0, cem.map_power(ident - g, t.nodes())(t))

    def test_antipode_squared_3(self):
        f = cem.antipode
        g = f @ f

        h = Map(lambda x : cem.map_power(ident - g, x.nodes() - 1)(x))
        m = (ident + f) @ h

        for t in trees[2:]: #Exclude the unit (and empty tree)
            self.assertEqual(0, m(t))

    def test_substitution_relations(self):
        b = Map(lambda x : x.nodes())
        b1 = Map(lambda x : x.nodes() ** 2)
        b2 = Map(lambda x : x.factorial() - 1)

        a = Map(lambda x : x.nodes() + 1)
        a1 = Map(lambda x : x.nodes() ** 2 + 1)
        a2 = Map(lambda x : x.factorial())

        m1 = (b1 ^ b2) ^ a
        m2 = b1 ^ (b2 ^ a)

        m3 = b ^ (a1 * a2)
        m4 = (b ^ a1) * (b ^ a2)

        m5 = (b ^ a) ** (-1)
        m6 = b ^ (a ** (-1))

        for t in trees[1:]:
            self.assertAlmostEqual(m1(t), m2(t), msg = repr(t))
            self.assertAlmostEqual(m3(t), m4(t), msg = repr(t))
            self.assertAlmostEqual(m5(t), m6(t), msg = repr(t))

    def test_omega(self):
        omegas_ = [1, -1/2, 1/6, 1/3, 0, -1/12, -1/6, -1/4]
        for i,t in enumerate(trees[1:]):
            self.assertAlmostEqual(omegas_[i], omega(t))

class RKTests(unittest.TestCase):
    def test_rk(self):
        pass

if __name__ == '__main__':
    unittest.main()