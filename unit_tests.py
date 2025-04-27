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

class TreeTests(unittest.TestCase):

    def test_repr(self):
        self.assertEqual(repr(T([[[]], []])), '[[[]], []]')
        self.assertEqual(repr(T([[[]], []]).as_forest()), '[[[]], []]')
        self.assertEqual(repr(T([[[]], []]).as_forest_sum()), '1*[[[]], []]')
        self.assertEqual(repr(T(None)), "∅")
        self.assertEqual(repr(Forest([])), "∅")
        self.assertEqual(repr(ForestSum([])), "0")

    def test_conversion(self):
        for t in trees:
            self.assertEqual(repr(t), repr(t.as_forest()), repr(t) + " " + repr(t.as_forest()))
            self.assertEqual("1*" + repr(t), repr(t.as_forest_sum()), repr(t) + " " + repr(t.as_forest_sum()))

    def test_add(self):
        t1 = T([]) + T([[],[]])
        t2 = ForestSum((
            (1, Forest([T([])])),
            (1, Forest([T([[],[]])]))
        ))
        self.assertEqual(t2, t1)

        t1 = T([]) - 2 * T([[], []])
        t2 = ForestSum((
            (1, Forest([T([])])),
            (-2, Forest([T([[], []])]))
        ))
        self.assertEqual(t2, t1)

        t1 = 1 + T([[], []])
        t2 = ForestSum((
            (1, Forest([T(None)])),
            (1, Forest([T([[], []])]))
        ))
        self.assertEqual(t2, t1)

        t1 = T([[], []]) + 2
        t2 = ForestSum((
            (1, Forest([T([[], []])])),
            (2, Forest([T(None)]))
        ))
        self.assertEqual(t2, t1)

        t1 = T([[], []]) + Forest([T([]), T([[]])])
        t2 = ForestSum((
            (1, Forest([T([[], []])])),
            (1, Forest([T([]), T([[]])]))
        ))
        self.assertEqual(t2, t1)

    def test_mul(self):
        t1 = T([]) * T([[], []])
        t2 = Forest([T([]), T([[],[]])])
        self.assertEqual(t2, t1)

        t1 = T([]) * T(None) * T(None)
        t2 = T([])
        self.assertEqual(t1, t2)

        t1 = T([[]]) * Forest([T([]), T([[],[]])])
        t2 = Forest([T([[]]), T([]), T([[],[]])])
        self.assertEqual(t1, t2)

        t1 = (T([]) - T([[]])) * T([])
        t2 = T([]) * T([]) - T([[]]) * T([])
        self.assertEqual(t1,t2)

    def test_pow_tree(self):
        t1 = T([]) ** 3
        t2 = T([]) * T([]) * T([])
        self.assertEqual(t1, t2)

        t1 = T(None) ** 3
        self.assertEqual(1, t1)

        t1 = T([]) ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(ValueError):
            t1 ** 1.5

    def test_pow_forest(self):
        t1 = Forest([T([]), T(None)]) ** 3
        t2 = T([]) * T([]) * T([])
        self.assertEqual(t1, t2)

        t1 = T(None).as_forest() ** 3
        self.assertEqual(1, t1)

        t1 = T([]).as_forest() ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(ValueError):
            t1 ** 1.5

    def test_pow_forest_sum(self):
        t1 = (T([]) - 2 * T([[]])) ** 2
        t2 = T([]) * T([]) - 4 * T([]) * T([[]]) + 4 * T([[]]) * T([[]])
        self.assertEqual(t1, t2)

        t1 = T([]).as_forest_sum() ** 0
        self.assertEqual(1, t1)

        with self.assertRaises(ValueError):
            t1 ** -1
        with self.assertRaises(ValueError):
            t1 ** 1.5

    def test_equality(self):
        self.assertEqual(T([[],[[]]]), T([[[]],[]]))
        self.assertEqual(T([[], [[]]]).as_forest(), T([[[]], []]))
        self.assertEqual(T([[],[[],[]]]), T([[[],[]], []]))
        self.assertEqual(T([[[]],[],[]]), T([[],[[]],[]]))
        self.assertEqual(T([[[]], [], []]), T([[], [], [[]]]))
        self.assertEqual(Forest([T([[]]), T([]), T([[],[]])]), Forest([T([]), T([[],[]]), T([[]])]))
        self.assertEqual(T([]) - T([[]]), -T([[]]) + T([]))
        self.assertEqual(ForestSum([(1, T([[]]).as_forest()), (1, T([]).as_forest()), (-1, T([]).as_forest())]), T([[]]).as_forest_sum())

    def test_equality_2(self):
        self.assertEqual(
            2*T([[]]) * T([]),
            2*T([]) * T([[]])
        )

        self.assertEqual(
            -T([]) * T([]) * T([]) + 2 * T([[]]) * T([]) - T([[], []]),
            - T([[], []]) + 2 * T([]) * T([[]]) -T([]) * T([]) * T([])
        )

    def test_hash(self):
        self.assertEqual(hash(T([[],[[]]])), hash(T([[[]],[]])))
        self.assertEqual(hash(T([[],[[],[]]])), hash(T([[[],[]], []])))
        self.assertEqual(hash(T([[[]],[],[]])), hash(T([[],[[]],[]])))
        self.assertEqual(hash(T([[[]], [], []])), hash(T([[], [], [[]]])))
        self.assertEqual(hash(Forest([T([[]]), T([]), T([[],[]])])), hash(Forest([T([]), T([[],[]]), T([[]])])))
        self.assertEqual(hash(T([]) - T([[]])), hash(-T([[]]) + T([])))
        self.assertEqual(hash(T([[]]) + T([]) - T([])), hash(T([[]]).as_forest_sum()))

    def test_mixed_arithmetic(self):
        t0 = T(None)
        t1 = T([])
        t2 = T([[]])
        t3 = T([[], []])

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
            self.assertEqual(n, t.nodes(), repr(t) + " T")
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
            self.assertEqual(f, t.factorial(), repr(t) + " T")
            self.assertEqual(f, t.as_forest().factorial(), repr(t) + " Forest")
            self.assertEqual(f, t.as_forest_sum().factorial(), repr(t) + " ForestSum")

class BCKTests(unittest.TestCase):

    def test_coproduct(self):
        trees_ = [
            T([]),
            T([[]]),
            T([[],[]]),
            T([[[]]])
        ]
        true_coproducts_ = [
            T([]) @ T() + T() @ T([]),
            T([[]]) @ T() + T() @ T([[]]) + T([]) @ T([]),
            T([[],[]]) @ T() + T() @ T([[],[]]) + 2 * T([]) @ T([[]]) + T([]) * T([]) @ T([]),
            T([[[]]]) @ T() + T() @ T([[[]]]) + T([[]]) @ T([]) + T([]) @ T([[]])
        ]
        for t, c in zip(trees_, true_coproducts_):
            self.assertEqual(c, bck.coproduct(t))

    def test_antipode(self):
        antipodes = [
            1*T(None),
            -T([]),
            T([]) * T([]) - T([[]]),
            -T([]) * T([]) * T([]) + 2 * T([[]]) * T([]) - T([[],[]]),
            -T([]) * T([]) * T([]) + 2 * T([[]]) * T([]) - T([[[]]]),
            T([]) * T([]) * T([]) * T([]) - 3 * T([[]]) * T([]) * T([]) + 3 * T([[],[]]) * T([]) - T([[],[],[]]),
            T([]) * T([]) * T([]) * T([]) - 3 * T([[]]) * T([]) * T([]) + T([[],[]]) * T([]) + T([[]]) * T([[]]) + T([[[]]]) * T([]) - T([[],[[]]])
        ]

        for t, s in zip(trees[:7], antipodes):
            self.assertEqual(s, bck.antipode(t), repr(t) + " T")
            self.assertEqual(s, bck.antipode(t.as_forest()), repr(t) + " Forest")
            self.assertEqual(s, bck.antipode(t.as_forest_sum()), repr(t) + " ForestSum")

    def test_antipode_property(self):
        m1 = bck.antipode * ident
        m2 = ident * bck.antipode
        for t in trees:
            self.assertEqual(bck.counit(t), m1(t))
            self.assertEqual(bck.counit(t), m2(t))

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
        m1 = (S * S) * S
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

    def test_product_scalar(self):
        m1 = ident ^ 2
        m2 = 2 ^ ident
        m3 = ident * 2
        m4 = 2 * ident
        for t in trees:
            self.assertEqual(m1(t), 2 * t)
            self.assertEqual(m2(t), 2 * t)
            self.assertEqual(m3(t), 2 * t)
            self.assertEqual(m4(t), 2 * t)

    def test_inverse_identity(self):
        a = Map(lambda x : 1 if x == T(None) or x == T([]) else 0) ** (-1)
        for t in trees:
            self.assertEqual((-1)**t.nodes(), a(t), repr(t))

class CEMTests(unittest.TestCase):

    def test_coproduct(self):
        trees_ = [
            T([]),
            T([[]]),
            T([[[]]]),
            T([[[[]]]]),
            T([[[[[]]]]]),
            T([[],[],[]])
        ]
        true_coproducts_ = [
            T([]) @ T([]),
            T([[]]) @ T([]) + T([]) @ T([[]]),
            T([[[]]]) @ T([]) + T([]) @ T([[[]]]) + 2 * T([[]]) @ T([[]]),
            T([[[[]]]]) @ T([]) + T([]) @ T([[[[]]]]) + 2 * T([[[]]]) @ T([[]]) + 3 * T([[]]) @ T([[[]]]) + T([[]]) * T([[]]) @ T([[]]),
            T([[[[[]]]]]) @ T([]) + T([]) @ T([[[[[]]]]]) + 2 * T([[[[]]]]) @ T([[]]) + 3 * T([[[]]]) @ T([[[]]]) + 4 * T([[]]) @ T([[[[]]]]) + 3 * T([[]]) * T([[]]) @ T([[[]]]) + 2 * T([[[]]]) * T([[]]) @ T([[]]),
            T([[],[],[]]) @ T([]) + T([]) @ T([[],[],[]]) + 3*T([[],[]]) @ T([[]]) + 3 * T([[]]) @ T([[],[]])
        ]
        for t, c in zip(trees_, true_coproducts_):
            self.assertEqual(c, cem.coproduct(t), msg = repr(t))

    def test_antipode(self):
        trees_ = [
            T([]),
            T([[]]),
            T([[],[]]),
            T([[[]]])
        ]
        antipodes_ = [
            T([]),
            -T([[]]),
            -T([[],[]]) + 2 * T([[]])**2,
            -T([[[]]]) + 2 * T([[]])**2
        ]
        for t, a in zip(trees_, antipodes_):
            self.assertEqual(a, cem.antipode(t))

    def test_antipode_property(self):
        m1 = cem.antipode ^ ident
        m2 = ident ^ cem.antipode
        for t in trees[1:]:
            self.assertEqual((cem.counit(t) * T([])), m1(t), repr(t))
            self.assertEqual((cem.counit(t) * T([])), m2(t), repr(t))

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

        for t in trees[2:]: #Exclude the unit (and empty T)
            self.assertEqual(0, m(t))

    def test_substitution_relations(self):
        b = Map(lambda x : x.nodes())
        b1 = Map(lambda x : x.nodes() ** 2)
        b2 = Map(lambda x : x.factorial() - 1 if x != Tree([]) else 1)

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

    def test_log_exp(self):
        m1 = Map(lambda x : x.factorial())
        m2 = m1.exp().log()
        m3 = m1.log().exp()
        for t in trees:
            self.assertAlmostEqual(m1(t), m2(t))
            self.assertAlmostEqual(m1(t), m3(t))

class RKTests(unittest.TestCase):
    def test_rk(self):
        pass #TODO

class TensorProductSumTests(unittest.TestCase):
    def test_tensor(self):
        pass #TODO

if __name__ == '__main__':
    unittest.main()
