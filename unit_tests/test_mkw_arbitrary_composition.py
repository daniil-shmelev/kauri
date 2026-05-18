# Copyright 2026 Daniil Shmelev
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
"""Stress tests for arbitrary composition of LB characters via
:func:`kauri.mkw.map_product`.

Under the paper's forest coproduct + basis-aware character evaluation,
convolution is genuinely associative on the MKW Hopf algebra.  These
tests exercise that on every planar tree up to order 5, across multiple
random seeds and named CF method characters."""
import random
import unittest

from fractions import Fraction as F

from kauri import (
    lie_euler, lie_midpoint, cfree_rk3, cfree_rk4, PlanarTree,
    Map,
)
from kauri.gentrees import planar_trees_up_to_order
import kauri.mkw as mkw
from kauri.generic_algebra import mkw_base_char_func


def _random_tree_char(seed, max_order=5):
    """Random shuffle-symmetric base character on planar trees."""
    rng = random.Random(seed)
    values = {None: F(1)}
    for t in planar_trees_up_to_order(max_order):
        if t.list_repr is not None and t.list_repr not in values:
            values[t.list_repr] = F(rng.randint(-50, 50), 1)

    def tree_fn(tree):
        return values[tree.list_repr]

    m = Map(mkw_base_char_func(tree_fn), extension="shuffle")
    m._mkw_basis_aware = True
    return m


class RandomAssociativityTests(unittest.TestCase):
    """For arbitrary shuffle-symmetric random characters f, g, h, the
    convolution satisfies ``((f*g)*h)(tau) == (f*(g*h))(tau)`` on every
    planar tree."""

    def _check_associativity(self, f, g, h, max_order):
        fg = mkw.map_product(f, g)
        gh = mkw.map_product(g, h)
        fg_h = mkw.map_product(fg, h)
        f_gh = mkw.map_product(f, gh)
        for t in planar_trees_up_to_order(max_order):
            self.assertEqual(fg_h(t), f_gh(t),
                msg=f"Associativity failed at {t.list_repr}: "
                    f"((f*g)*h)={fg_h(t)} vs (f*(g*h))={f_gh(t)}")

    def test_associativity_random_seed_0(self):
        f = _random_tree_char(0)
        g = _random_tree_char(1)
        h = _random_tree_char(2)
        self._check_associativity(f, g, h, max_order=5)

    def test_associativity_random_seed_42(self):
        f = _random_tree_char(42)
        g = _random_tree_char(43)
        h = _random_tree_char(44)
        self._check_associativity(f, g, h, max_order=5)

    def test_associativity_random_seed_100(self):
        f = _random_tree_char(100)
        g = _random_tree_char(101)
        h = _random_tree_char(102)
        self._check_associativity(f, g, h, max_order=5)


class CFMethodCompositionTests(unittest.TestCase):
    """Arbitrary compositions of CF method LB characters."""

    def test_three_fold_cf_composition_associates(self):
        a = lie_euler.lb_character()
        b = lie_midpoint.lb_character()
        c = cfree_rk3.lb_character()

        ab_c = mkw.map_product(mkw.map_product(a, b), c)
        a_bc = mkw.map_product(a, mkw.map_product(b, c))

        for t in planar_trees_up_to_order(5):
            self.assertAlmostEqual(ab_c(t), a_bc(t), places=10,
                msg=f"3-fold CF composition non-associative at {t.list_repr}")

    def test_four_fold_cf_composition_associates(self):
        a = lie_euler.lb_character()
        b = lie_midpoint.lb_character()
        c = cfree_rk3.lb_character()
        d = cfree_rk4.lb_character()

        left  = mkw.map_product(mkw.map_product(mkw.map_product(a, b), c), d)
        mid   = mkw.map_product(mkw.map_product(a, b), mkw.map_product(c, d))
        right = mkw.map_product(a, mkw.map_product(b, mkw.map_product(c, d)))

        for t in planar_trees_up_to_order(5):
            v_left = left(t)
            v_mid = mid(t)
            v_right = right(t)
            self.assertAlmostEqual(v_left, v_mid, places=10,
                msg=f"4-fold parenthesisation 1 vs 2 differs at {t.list_repr}")
            self.assertAlmostEqual(v_mid, v_right, places=10,
                msg=f"4-fold parenthesisation 2 vs 3 differs at {t.list_repr}")

    def test_counit_is_left_and_right_identity(self):
        """counit * alpha = alpha = alpha * counit on every planar tree."""
        alpha = cfree_rk3.lb_character()
        left = mkw.map_product(mkw.counit, alpha)
        right = mkw.map_product(alpha, mkw.counit)
        for t in planar_trees_up_to_order(5):
            self.assertAlmostEqual(left(t), alpha(t), places=10,
                msg=f"counit * alpha != alpha at {t.list_repr}")
            self.assertAlmostEqual(right(t), alpha(t), places=10,
                msg=f"alpha * counit != alpha at {t.list_repr}")

    def test_convolution_inverse_via_antipode(self):
        """f^{-1} * f = counit on every planar tree."""
        alpha = cfree_rk3.lb_character()
        alpha_inv = mkw.map_power(alpha, -1)
        composed = mkw.map_product(alpha_inv, alpha)
        for t in planar_trees_up_to_order(4):
            expected = 1 if t.list_repr is None else 0
            self.assertAlmostEqual(composed(t), expected, places=10,
                msg=f"f^-1 * f != counit at {t.list_repr}: got {composed(t)}")


if __name__ == "__main__":
    unittest.main()
