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
    def test_elementary_weights(self):
        # Test using an RK method of order 4
        scheme = rk4
        rk_weights = scheme.elementary_weights_map()

        for t in trees:
            self.assertAlmostEqual(exact_weights(t), rk_weights(t))

    def test_order(self):
        schemes = [heun_rk2, heun_rk3, rk4, nystrom_rk5]
        orders = [2,3,4,5]

        for scheme, order in zip(schemes, orders):
            self.assertEqual(order, scheme.order())

    def test_symbolic_weight(self):
        t = Tree([[], []])
        self.assertEqual("a10**2*b1 + b2*(a20 + a21)**2", str(rk_symbolic_weight(t, 3, True)))

    def test_order_cond(self):
        t = Tree([[], []])
        self.assertEqual("a10**2*b1 + b2*(a20 + a21)**2 - 1/3", str(rk_order_cond(t, 3, True)))

    def test_inverse(self):
        method = rk4
        inv_method = method ** (-1)
        id = method * inv_method
        m = id.elementary_weights_map()

        for t in trees_up_to_order(5):
            self.assertAlmostEqual(bck.counit(t), m(t))
    #
    # def test_add(self):
    #     method1 = rk4
    #     method2 = euler
    #     sum_method = method1 + method2
    #
    #     m1 = method1.elementary_weights_map() + method2.elementary_weights_map()
    #     m2 = sum_method.elementary_weights_map()
    #
    #     for t in trees_up_to_order(5):
    #         if t == Tree(None):
    #             continue
    #         self.assertAlmostEqual(m1(t), m2(t), msg = repr(t))