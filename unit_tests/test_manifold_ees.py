"""Tests for Phases A–D: PlanarTree extensions, RK on ordered trees,
CF methods, and symbolic order-condition generation."""

import unittest
import sympy

from kauri.trees import PlanarTree, NoncommutativeForest, OrderedForest, EMPTY_PLANAR_TREE
from kauri.maps import exact_weights, Map
from kauri.gentrees import planar_trees_of_order
from kauri.rk import rk_symbolic_weight, rk_order_cond, RK
from kauri.cf import CFMethod
from kauri import EES25, EES27, rk4, euler


# ── Phase A: PlanarTree.factorial / sigma / density ──────────────────────

class TestPhaseA(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(PlanarTree(None).factorial(), 1)
        self.assertEqual(PlanarTree([]).factorial(), 1)
        self.assertEqual(PlanarTree([[]]).factorial(), 2)
        self.assertEqual(PlanarTree([[], []]).factorial(), 3)
        self.assertEqual(PlanarTree([[[]]]).factorial(), 6)

    def test_sigma_always_one(self):
        for n in range(5):
            for t in planar_trees_of_order(n):
                self.assertEqual(t.sigma(), 1, msg=repr(t.list_repr))

    def test_density(self):
        self.assertAlmostEqual(PlanarTree([[]]).density(), 1.0)
        self.assertAlmostEqual(PlanarTree([[], []]).density(), 0.5)

    def test_ncf_factorial(self):
        t1 = PlanarTree([])
        t2 = PlanarTree([[]])
        f = NoncommutativeForest((t1, t2))
        self.assertEqual(f.factorial(), 1 * 2)

    def test_exact_weights_on_planar(self):
        self.assertAlmostEqual(exact_weights(PlanarTree([[]])), 0.5)
        self.assertAlmostEqual(exact_weights(PlanarTree([[], []])), 1. / 3)
        self.assertAlmostEqual(exact_weights(PlanarTree([[[]]])), 1. / 6)


# ── Phase B: RK on ordered trees ────────────────────────────────────────

class TestPhaseB(unittest.TestCase):
    def test_symbolic_weight_on_planar(self):
        t = PlanarTree([[], []])
        w = rk_symbolic_weight(t, 2, True)
        self.assertEqual(str(w), "a10**2*b1")

    def test_order_cond_on_planar(self):
        t = PlanarTree([[], []])
        c = rk_order_cond(t, 2, True)
        self.assertEqual(str(c), "a10**2*b1 - 1/3")

    def test_rk4_planar_order(self):
        self.assertEqual(rk4.planar_order(), 4)

    def test_euler_planar_order(self):
        self.assertEqual(euler.planar_order(), 1)

    def test_ees25_planar(self):
        rk = EES25(0.1)
        self.assertEqual(rk.planar_order(), 2)
        self.assertEqual(rk.planar_antisymmetric_order(), 5)

    def test_ees27_planar(self):
        rk = EES27(0.1)
        self.assertEqual(rk.planar_order(), 2)
        self.assertEqual(rk.planar_antisymmetric_order(), 7)


# ── Phase C: CFMethod ───────────────────────────────────────────────────

class TestPhaseC(unittest.TestCase):
    def _ees25_params(self):
        a = [[0, 0, 0], [0.5, 0, 0], [0, 1, 0]]
        b = [0.25, 0.5, 0.25]
        return a, b

    def test_single_exponential_matches_rk(self):
        """J=1 CF should give the same planar ORDER as the underlying RK (both
        evaluate tree character values, which agree).  The ANTISYMMETRIC
        order can differ: ``RK.planar_antisymmetric_order`` uses the NCK
        convolution for the symmetry defect, whereas
        ``CFMethod.planar_antisymmetric_order`` uses the MKW convolution
        (the Lie-group convention).  EES25 was designed to satisfy NCK's
        antisymmetric order 5; under MKW it is 3."""
        a, b = self._ees25_params()
        cf = CFMethod(a, [b])
        self.assertEqual(cf.planar_order(), 2)
        self.assertEqual(cf.planar_antisymmetric_order(), 3)

    def test_projected_rk(self):
        a, b = self._ees25_params()
        betas = [[0.25, 0, 0], [0, 0.5, 0], [0, 0, 0.25]]
        cf = CFMethod(a, betas)
        rk = cf.projected_rk()
        self.assertEqual(rk.order(), 2)

    def test_lb_character_order_0_and_1(self):
        """LB character should satisfy alpha(empty)=1 and alpha(bullet)=sum(b) regardless of J."""
        a, b = self._ees25_params()
        betas = [[0.25, 0, 0], [0, 0.5, 0], [0, 0, 0.25]]
        cf = CFMethod(a, betas)
        alpha = cf.lb_character()
        self.assertAlmostEqual(alpha(EMPTY_PLANAR_TREE), 1.0)
        self.assertAlmostEqual(alpha(PlanarTree([])), 1.0)


# ── Phase D: Symbolic order conditions ──────────────────────────────────

class TestPhaseD(unittest.TestCase):
    def test_symbolic_lb_character_j1(self):
        from kauri.manifold_ees import symbolic_cf_params, symbolic_lb_character
        a, betas = symbolic_cf_params(3, 1, explicit=True)
        t = PlanarTree([[]])
        val = symbolic_lb_character(t, a, betas, 3, 1)
        # Should match rk_symbolic_weight with the same symbols
        expected = rk_symbolic_weight(t, 3, explicit=True)
        # Substitute: beta00->b0, beta01->b1, beta02->b2
        mapping = {
            sympy.Symbol('beta00'): sympy.Symbol('b0'),
            sympy.Symbol('beta01'): sympy.Symbol('b1'),
            sympy.Symbol('beta02'): sympy.Symbol('b2'),
        }
        self.assertEqual(sympy.expand(val.subs(mapping) - expected), 0)

    def test_forward_conditions_count(self):
        from kauri.manifold_ees import generate_conditions
        result = generate_conditions(2, 3, s=3, J=1, explicit=True)
        # Order 1: 1 tree -> 1 condition; Order 2: 1 tree -> 1 condition
        self.assertEqual(len(result['forward']), 2)

    def test_ees25_satisfies_conditions(self):
        """EES25 was designed under the NCK convention to have
        antisymmetric order 5; under the MKW (Lie-group) convention
        used by :func:`generate_conditions`, its antisymmetric order is
        3 — still a valid forward-order-2 / antisymmetric-order-3 method."""
        from kauri.manifold_ees import generate_conditions, verify_conditions
        result = generate_conditions(2, 3, s=3, J=1, explicit=True)
        subs = {
            sympy.Symbol('a10'): sympy.Rational(1, 2),
            sympy.Symbol('a20'): 0,
            sympy.Symbol('a21'): 1,
            sympy.Symbol('beta00'): sympy.Rational(1, 4),
            sympy.Symbol('beta01'): sympy.Rational(1, 2),
            sympy.Symbol('beta02'): sympy.Rational(1, 4),
        }
        ok, idx, resid = verify_conditions(result['all'], subs)
        self.assertTrue(ok, msg=f"Condition {idx} failed: {resid}")

    def test_mathematica_export(self):
        from kauri.manifold_ees import generate_conditions, mathematica_export
        result = generate_conditions(1, 1, s=2, J=1, explicit=True)
        code = mathematica_export(result['forward'])
        self.assertIn("== 0", code)

    def test_groebner_basis(self):
        from kauri.manifold_ees import generate_conditions, groebner_basis
        result = generate_conditions(1, 1, s=2, J=1, explicit=True)
        if result['forward']:
            gb = groebner_basis(result['forward'])
            self.assertIsNotNone(gb)


# ── EES character verification ─────────────────────────────────────────

class TestVerifyEESCharacter(unittest.TestCase):
    def test_counit_satisfies_ees(self):
        """The counit satisfies the EES (odd) condition up to order 5."""
        from kauri.manifold_ees import verify_ees_character
        counit = Map(lambda tree: 1 if tree == EMPTY_PLANAR_TREE else 0)
        self.assertTrue(verify_ees_character(counit, 5))

    def test_constant_map_fails_ees(self):
        """A constant map violates the EES condition."""
        from kauri.manifold_ees import verify_ees_character
        self.assertFalse(verify_ees_character(Map(lambda tree: 1), 3))


if __name__ == '__main__':
    unittest.main()
