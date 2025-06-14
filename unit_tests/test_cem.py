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
        g = f & f
        for t in trees[1:]:
            self.assertEqual(t, g(t))

    def test_antipode_squared_2(self):
        f = cem.antipode
        g = f & f

        for t in trees[1:]:
            self.assertEqual(0, cem.map_power(ident - g, t.nodes())(t))

    def test_antipode_squared_3(self):
        f = cem.antipode
        g = f & f

        h = Map(lambda x : cem.map_power(ident - g, x.nodes() - 1)(x))
        m = (ident + f) & h

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

    def test_type_error(self):
        with self.assertRaises(TypeError):
            cem.coproduct('s')
        with self.assertRaises(TypeError):
            cem.antipode('s')
        with self.assertRaises(TypeError):
            cem.counit('s')