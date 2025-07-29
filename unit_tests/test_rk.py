# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

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
        methods = [euler, heun_rk2, midpoint, kutta_rk3, heun_rk3,
                   ralston_rk3, rk4, ralston_rk4, nystrom_rk5, backward_euler,
                   implicit_midpoint, crank_nicolson, gauss6, radau_iia, lobatto6]
        orders = [1, 2, 2, 3, 3, 3, 4, 4, 5, 1, 2, 2, 6, 5, 6]

        for m, ord in zip(methods, orders):
            self.assertEqual(ord, m.order(), msg=m.name)

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

    def test_ees25_order(self):
        rk = EES25(0.1)
        self.assertEqual(2, rk.order())
        self.assertEqual(5, rk.antisymmetric_order())

    def test_ees27_order(self):
        rk = EES27(0.1)
        self.assertEqual(2, rk.order())
        self.assertEqual(7, rk.antisymmetric_order())