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
        g = f & f
        for t in trees:
            self.assertEqual(t, g(t))

    def test_antipode_squared_2(self):
        f = bck.antipode
        g = f & f

        for t in trees[1:]:
            self.assertEqual(0, ((ident - g) ** t.nodes())(t))

    def test_antipode_squared_3(self):
        f = bck.antipode
        g = f & f

        h = Map(lambda x: ((ident - g) ** (x.nodes() - 1))(x))
        m = (ident + f) & h

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

    def test_type_error(self):
        with self.assertRaises(TypeError):
            bck.coproduct('s')
        with self.assertRaises(TypeError):
            bck.antipode('s')
        with self.assertRaises(TypeError):
            bck.counit('s')