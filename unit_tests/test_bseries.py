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
import sympy as sp
import math

class BCKTests(unittest.TestCase):

    def test_elementary_differentials(self):
        y1 = sp.symbols('y1')
        y = sp.Matrix([y1])
        f = sp.Matrix([y1**2])
        trees = [
            T(None),
            T([]),
            T([[]]),
            T([[],[]]),
            T([[[]]]),
            T([[[]],[]])
        ]
        diffs = [
            sp.Matrix([y1]),
            sp.Matrix([y1**2]),
            sp.Matrix([2 * y1 ** 3]),
            sp.Matrix([2 * y1 ** 4]),
            sp.Matrix([4 * y1 ** 4]),
            sp.Matrix([4* y1 ** 5])
        ]
        for t, d in zip(trees, diffs):
            self.assertEqual(d, elementary_differential(t, f, y))

    def test_elementary_differentials_2(self):
        y1, y2 = sp.symbols('y1 y2')
        y = sp.Matrix([y1, y2])
        f = sp.Matrix([y1 * y2, y1 + y2])
        trees = [
            T(None),
            T([]),
            T([[]]),
            T([[],[]])
        ]
        diffs = [
            sp.Matrix([y1, y2]),
            sp.Matrix([y1 * y2, y1 + y2]),
            sp.Matrix([y1 * y2 ** 2 + y1 * (y1 + y2), y1*y2 + y1 + y2]),
            sp.Matrix([2 * y1 * y2 * (y1 + y2), 0])
        ]
        for t, d in zip(trees, diffs):
            self.assertEqual(d, elementary_differential(t, f, y))

    def test_exp(self):
        y1 = sp.symbols('y1')
        y = sp.Matrix([y1])
        f = sp.Matrix([y1])
        bs = BSeries(y, f, exact_weights, 8)
        expr = sp.Poly(bs.symbolic_expr.subs(bs.y[0], 1)[0, 0], bs.h)
        c = expr.all_coeffs()[::-1]
        for i, c_ in enumerate(c):
            self.assertAlmostEqual(1 / math.factorial(i), c_)

    def test_y_sq(self):
        y1 = sp.symbols('y1')
        y = sp.Matrix([y1])
        f = sp.Matrix([y1 ** 2])
        bs = BSeries(y, f, exact_weights, 8)
        expr = sp.Poly(bs.symbolic_expr.subs(bs.y[0], 1)[0, 0], bs.h)
        c = expr.all_coeffs()
        for c_ in c:
            self.assertAlmostEqual(1, c_)

    def test_extra_variables(self):
        y1, y2 = sp.symbols('y1 y2')
        y = sp.Matrix([y1])
        f = sp.Matrix([y1 * y2])

        with self.assertRaises(ValueError):
            bs = BSeries(y, f, exact_weights, 1)

    def test_misspecified(self):
        y1, y2 = sp.symbols('y1 y2')
        y = sp.Matrix([y1, y2])
        f = sp.Matrix([[y1 * y2, y2],[y1, y2]])

        with self.assertRaises(ValueError):
            bs = BSeries(y, f, exact_weights, 1)

    def test_inverse(self):
        y1 = sp.symbols('y1')
        y = sp.Matrix([y1])
        f = sp.Matrix([y1 ** 2])
        bs1 = BSeries(y, f, exact_weights, 5)
        bs2 = BSeries(y, f, exact_weights & bck.antipode, 5)
        bs3 = BSeries(y, f, bs2.weights * bs1.weights, 5)
        expr = sp.Poly(bs3.symbolic_expr.subs(bs3.y[0], 1)[0, 0], bs3.h)
        c = expr.all_coeffs()
        self.assertAlmostEqual(1., c[-1])
        for c_ in c[:-1]:
            self.assertAlmostEqual(0., float(c_))