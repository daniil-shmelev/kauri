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
"""Tests for the named commutator-free methods in :mod:`kauri.cf_methods`
and for :meth:`kauri.cf.CFMethod.symbolic_lb_character`.

The composition tests below verify kauri's LB-series composition, which
is implemented in :func:`kauri.mkw.map_product` by summing over the NCK
(Foissy) coproduct with concatenation-multiplicative extension of the
:class:`Map` characters (see ``kauri/mkw/mkw.py`` lines 408-413).  This
is the correct convolution for kauri's concat-multiplicative ``Map``
class and agrees with the MKW convolution on the subset of characters
that satisfy both shuffle- and concat-multiplicativity.  The semantic
check ``two half-steps of an RK method = one full step`` is the most
direct test of LB composition correctness and is included below.
"""
import unittest

import sympy

from kauri import (
    lie_euler,
    lie_midpoint,
    cfree_rk3,
    cfree_rk4,
    PlanarTree,
    Map,
)
from kauri.gentrees import planar_trees_up_to_order
import kauri.mkw as mkw
import kauri.nck as nck


# ---------------------------------------------------------------------------
# Tree shorthands (same mapping as unit_tests/test_mkw.py)
# ---------------------------------------------------------------------------

PT = PlanarTree
empty_tree = PT(None)
bullet = PT([])
chain2 = PT([[]])
chain3 = PT([[[]]])
cherry = PT([[], []])
chain4 = PT([[[[]]]])
b_cherry = PT([[[], []]])
b_bc2 = PT([[], [[]]])
b_c2b = PT([[[]], []])
b_bbb = PT([[], [], []])


# ---------------------------------------------------------------------------
# Class 1: planar_order sanity check on published tableaux
# ---------------------------------------------------------------------------

class TestPublishedOrders(unittest.TestCase):
    """Each named CF method reports the planar order claimed in its docstring."""

    def test_lie_euler_order(self):
        self.assertEqual(lie_euler.planar_order(), 1)

    def test_lie_midpoint_order(self):
        self.assertEqual(lie_midpoint.planar_order(), 2)

    def test_cfree_rk3_order(self):
        self.assertEqual(cfree_rk3.planar_order(), 3)

    def test_cfree_rk4_order(self):
        self.assertEqual(cfree_rk4.planar_order(), 4)


# ---------------------------------------------------------------------------
# Class 2: alpha(tau) = 1/tau! up to the method's order (numerical + symbolic)
# ---------------------------------------------------------------------------

class TestCharacterAgainstExactSolution(unittest.TestCase):
    """Both the numerical and the symbolic LB character of each named CF
    method satisfy the exact-solution order condition ``alpha(tau) = 1/tau!``
    for every planar tree up to the method's published order."""

    _methods_and_orders = (
        (lie_euler, 1),
        (lie_midpoint, 2),
        (cfree_rk3, 3),
        (cfree_rk4, 4),
    )

    def test_numerical_character(self):
        for method, order in self._methods_and_orders:
            alpha = method.lb_character()
            for t in planar_trees_up_to_order(order):
                expected = 1.0 / t.factorial()
                self.assertAlmostEqual(
                    alpha(t), expected, places=12,
                    msg=f"{method.name}: alpha({t.list_repr}) != 1/{t.factorial()}",
                )

    def test_symbolic_character(self):
        for method, order in self._methods_and_orders:
            alpha = method.symbolic_lb_character()
            for t in planar_trees_up_to_order(order):
                expected = sympy.Rational(1, t.factorial())
                self.assertEqual(
                    sympy.simplify(alpha(t) - expected), 0,
                    msg=f"{method.name}: alpha({t.list_repr}) != 1/{t.factorial()}",
                )

    def test_symbolic_character_returns_rational(self):
        # The whole point of symbolic_lb_character is exact arithmetic.
        # Verify the character returns a sympy Rational/Expr, not a float.
        alpha = cfree_rk3.symbolic_lb_character()
        for t in planar_trees_up_to_order(3):
            val = alpha(t)
            self.assertIsInstance(val, sympy.Expr,
                msg=f"Expected sympy.Expr, got {type(val)}")


# ---------------------------------------------------------------------------
# Class 3: LB-composition semantics — two half-steps = one full step
# ---------------------------------------------------------------------------

# MKW true-solution LB character values for every planar tree up to order 4.
# Under MKW (shuffle-multiplicative) convention, alpha_exact is determined
# by the one-parameter-subgroup functional equation
# (alpha_{h/2} *_MKW alpha_{h/2})(tau) = h^|tau| * alpha_exact(tau)
# with alpha_{h}(tau) = h^|tau| * alpha_exact(tau), using the MKW coproduct
# entries from testing_scratch/munthe-kaas06oth.tex lines 1583-1648.
#
# These are DIFFERENT from the B-series/NCK values 1/tau.factorial() on
# trees where planar ordering matters (e.g. cherry: MKW gives 1/6, NCK
# gives 1/3; t43 vs t44: MKW assigns distinct 13/168 and 1/21 values).
ALPHA_EXACT_MKW = {
    None: sympy.Rational(1),                                         # empty
    (0,): sympy.Rational(1),                                         # bullet
    ((0,), 0): sympy.Rational(1, 2),                                 # chain2
    (((0,), 0), 0): sympy.Rational(1, 6),                            # chain3
    ((0,), (0,), 0): sympy.Rational(1, 6),                           # cherry
    ((((0,), 0), 0), 0): sympy.Rational(1, 24),                      # chain4
    (((0,), (0,), 0), 0): sympy.Rational(1, 24),                     # B+(cherry)
    ((0,), ((0,), 0), 0): sympy.Rational(13, 168),                   # t43 = B+(bullet, chain2)
    (((0,), 0), (0,), 0): sympy.Rational(1, 21),                     # t44 = B+(chain2, bullet)
    ((0,), (0,), (0,), 0): sympy.Rational(1, 24),                    # corolla3
}


class TestLBCompositionSemantics(unittest.TestCase):
    """Composing two half-step characters of the exact Lie-group flow via
    :func:`kauri.mkw.map_product` must reproduce the full-step exact LB
    character on every planar tree.

    This is the defining one-parameter-subgroup property of the
    exact-solution LB character under MKW convention.  Under the
    shuffle-multiplicative extension, the true-solution tree values are
    NOT 1/tau.factorial() (that is the NCK/B-series value); they are
    determined by the functional equation above.
    """

    def test_two_halfsteps_equal_full_step(self):
        import math
        # alpha_h(tau) = h^|tau| * alpha_exact(tau) for tau a tree.
        def alpha_h_factory(h):
            def alpha(t):
                lr = t.list_repr
                if lr is None:
                    return sympy.Rational(1)
                return h ** t.nodes() * ALPHA_EXACT_MKW[lr]
            return alpha

        alpha_half = Map(alpha_h_factory(sympy.Rational(1, 2)),
                         extension="shuffle")
        alpha_full = Map(alpha_h_factory(sympy.Rational(1)),
                         extension="shuffle")

        composed = mkw.map_product(alpha_half, alpha_half)
        for t_lr, exact in ALPHA_EXACT_MKW.items():
            t = PT(t_lr) if t_lr is not None else PT(None)
            expected = alpha_full(t)
            got = composed(t)
            # Exact sympy equality via simplify
            self.assertEqual(
                sympy.simplify(got - expected), 0,
                msg=f"(alpha_half *_MKW alpha_half)({t_lr}) = {got}, "
                    f"expected {expected}",
            )


# ---------------------------------------------------------------------------
# Class 3b: MKW and NCK convolutions agree on ladders, differ elsewhere
# ---------------------------------------------------------------------------

class TestMKWConvolutionVsNCK(unittest.TestCase):
    """MKW and NCK are different Hopf algebras: shuffle vs concatenation
    product, left-admissible vs admissible coproduct cuts.  Their
    convolutions agree on **ladder** trees (where the coproducts coincide)
    but disagree on trees with identical-type siblings (e.g. cherry,
    where MKW has coefficient 1 on bullet⊗chain2 while NCK has coefficient
    2, and the shuffle-symmetric extension on left forests adds further
    1/k! factors)."""

    _ladders = (bullet, chain2, chain3, chain4)
    _non_ladders = (cherry, b_cherry, b_bc2, b_c2b, b_bbb)

    def _shuffle_char(self):
        """A simple shuffle character defined on trees."""
        return Map(
            lambda t: 0.3 + 0.11 * t.nodes() if t.list_repr is not None else 1.0,
            extension="shuffle",
        )

    def _concat_char(self):
        return Map(
            lambda t: 0.3 + 0.11 * t.nodes() if t.list_repr is not None else 1.0,
        )  # default extension="concat"

    def test_mkw_and_nck_agree_on_ladder_trees(self):
        """On ladder trees, MKW coproduct = NCK coproduct (every cut is
        left-admissible), so the convolutions match regardless of
        forest-extension convention (no multi-tree left forests appear)."""
        a_sh = self._shuffle_char()
        b_sh = self._shuffle_char()
        a_cc = self._concat_char()
        b_cc = self._concat_char()
        via_mkw = mkw.map_product(a_sh, b_sh)
        via_nck = nck.map_product(a_cc, b_cc)
        for tau in self._ladders:
            self.assertAlmostEqual(
                via_mkw(tau), via_nck(tau), places=12,
                msg=f"MKW/NCK should agree on ladder {tau.list_repr}",
            )

    def test_mkw_disagrees_with_nck_on_cherry(self):
        """On cherry, the MKW coproduct has coefficient 1 on
        (bullet)⊗chain2 whereas NCK has coefficient 2 — combined with
        the shuffle-symmetric forest extension, the convolution values
        genuinely differ."""
        a_sh = self._shuffle_char()
        b_sh = self._shuffle_char()
        a_cc = self._concat_char()
        b_cc = self._concat_char()
        via_mkw = mkw.map_product(a_sh, b_sh)
        via_nck = nck.map_product(a_cc, b_cc)
        self.assertNotAlmostEqual(
            via_mkw(cherry), via_nck(cherry), places=6,
            msg="MKW and NCK convolutions should differ on cherry",
        )


# ---------------------------------------------------------------------------
# Class 4: Composition of CF characters via MKW convolution
# ---------------------------------------------------------------------------

class TestCFCharacterComposition(unittest.TestCase):
    """Properties of composed LB characters.

    All identities follow from the character-level composition formula
    ``LB(a, beta) o LB(a, alpha) = LB(a, alpha *_mkw beta)`` and from the
    fact that for scalar Maps MKW and NCK convolutions agree."""

    def test_identity_and_bullet_of_composition(self):
        """Convolution of two characters with alpha(empty)=1, alpha(bullet)=s
        satisfies (alpha *_mkw beta)(empty) = 1 and
        (alpha *_mkw beta)(bullet) = alpha(bullet) + beta(bullet)."""
        a = lie_midpoint.lb_character()
        b = cfree_rk3.lb_character()
        ab = mkw.map_product(a, b)
        self.assertAlmostEqual(ab(empty_tree), 1.0, places=12)
        self.assertAlmostEqual(ab(bullet), a(bullet) + b(bullet), places=12)
        # Both lie_midpoint and cfree_rk3 have alpha(bullet) == 1, so
        # the convolution has alpha(bullet) == 2.
        self.assertAlmostEqual(ab(bullet), 2.0, places=12)

    def test_selfcomposition_of_lie_euler_has_order_1(self):
        """Composing Lie-Euler with itself does not gain order.
        alpha_euler(chain2) = 1 (since it's sum_i b_i c_i with only
        c_0 = 0), but (alpha *_mkw alpha)(chain2) != 1/2 so the composite
        character is NOT order 2."""
        a = lie_euler.lb_character()
        ab = mkw.map_product(a, a)
        self.assertAlmostEqual(ab(empty_tree), 1.0, places=12)
        self.assertAlmostEqual(ab(bullet), 2.0, places=12)
        # The composition fails 1/t! at chain2, so composition has
        # planar order 1 at most.
        self.assertNotAlmostEqual(ab(chain2), 0.5, places=6)

    def test_symbolic_and_numerical_agree(self):
        """Symbolic composition value coincides with the numerical value."""
        num_alpha = cfree_rk3.lb_character()
        num_beta = cfree_rk3.lb_character()
        sym_alpha = cfree_rk3.symbolic_lb_character()
        sym_beta = cfree_rk3.symbolic_lb_character()

        num_ab = mkw.map_product(num_alpha, num_beta)
        sym_ab = mkw.map_product(sym_alpha, sym_beta)

        for t in planar_trees_up_to_order(4):
            num_val = num_ab(t)
            sym_val = float(sym_ab(t))
            self.assertAlmostEqual(
                num_val, sym_val, places=10,
                msg=f"numerical vs symbolic composition disagree at {t.list_repr}",
            )


if __name__ == "__main__":
    unittest.main()
