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
            self.assertEqual((f & S)(t), f(S(t)))

    def test_compose_scalar(self):
        f = Map(lambda x : x.factorial())

        for t in trees:
            self.assertEqual(f(t), (bck.counit & f)(t))

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

    def test_add_scalar(self):
        m1 = ident + 2
        m2 = 2 + ident
        m3 = ident - 2
        m4 = 2 - ident
        for t in trees:
            self.assertEqual(m1(t), 2 + t)
            self.assertEqual(m2(t), 2 + t)
            self.assertEqual(m3(t), t - 2)
            self.assertEqual(m4(t), 2 - t)

    def test_inverse_identity(self):
        a = Map(lambda x : 1 if x == T(None) or x == T([]) else 0) ** (-1)
        for t in trees:
            self.assertEqual((-1)**t.nodes(), a(t), repr(t))
