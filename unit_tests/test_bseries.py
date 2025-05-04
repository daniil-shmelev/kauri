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