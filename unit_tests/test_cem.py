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
from kauri import Tree, Map, ident, omega, cem
from kauri import Tree as T
from math import comb
from fractions import Fraction
from kauri.cem.cem import coproduct_impl

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

    # --- Literature-derived tests ---
    # [CHV] P. Chartier, E. Hairer, G. Vilmart,
    #       "Algebraic Structures of B-series",
    #       Found. Comput. Math. 10 (2010), pp. 407-427.
    #       https://doi.org/10.1007/s10208-010-9065-1
    # [CEM] D. Calaque, K. Ebrahimi-Fard, D. Manchon,
    #       "Two interacting Hopf algebras of trees:
    #       A Hopf-algebraic approach to composition and substitution of B-series",
    #       Adv. Appl. Math. 47 (2011), pp. 282-308.
    #       https://arxiv.org/abs/0806.2238

    def test_coproduct_chv_order_3_fork(self):
        """Δ_CEM(Y) = Y⊗● + 2/⊗/ + ●⊗Y  (singleton-reduced).
        Reference: [CHV] eq. (11), line 3.
        """
        expected = (
            T([[],[]]) @ T([])
            + T([]) @ T([[],[]])
            + 2 * T([[]]) @ T([[]])
        )
        self.assertEqual(expected, cem.coproduct(T([[],[]])))

    def test_coproduct_chv_order_4_trident(self):
        """Δ_CEM for [[],[],[]].
        Reference: [CHV] eq. (17), [CEM] §3.
        """
        expected = (
            T([[],[],[]]) @ T([])
            + T([]) @ T([[],[],[]])
            + 3 * T([[],[]]) @ T([[]])
            + 3 * T([[]]) @ T([[],[]])
        )
        self.assertEqual(expected, cem.coproduct(T([[],[],[]])))

    def test_antipode_cem(self):
        """CEM antipode S satisfying m(S ⊗ id) ∘ Δ = η ∘ ε.
        Reference: [CEM] §4, [CHV] §5.
        """
        self.assertEqual(T([]), cem.antipode(T([])))
        self.assertEqual(-1 * T([[]]), cem.antipode(T([[]])))
        self.assertEqual(
            -T([[],[]]) + 2 * T([[]]) ** 2,
            cem.antipode(T([[],[]]))
        )
        self.assertEqual(
            -T([[[]]]) + 2 * T([[]]) ** 2,
            cem.antipode(T([[[]]]))
        )

    def test_substitution_associativity(self):
        """(a ★ b) ★ c = a ★ (b ★ c).
        Reference: [CHV] discussion after eq. (17).
        """
        a = Map(lambda x: x.nodes())
        b = Map(lambda x: x.factorial())
        c = Map(lambda x: x.nodes() ** 2)

        trees_ = [T([]), T([[]]), T([[],[]]), T([[[]]]), T([[],[],[]])]
        for t in trees_:
            self.assertAlmostEqual(
                ((a ^ b) ^ c)(t), (a ^ (b ^ c))(t), msg=repr(t))

    def test_substitution_explicit(self):
        """Explicit substitution law for trees up to order 3.
        Reference: [CHV] eq. (6).
        """
        a = Map(lambda x: x.nodes() if x.list_repr is not None else 0)
        b = Map(lambda x: x.factorial() if x.list_repr is not None else 0)

        dot = T([])
        line = T([[]])
        fork = T([[],[]])

        # (b ★ a)(●) = a(●)b(●)
        self.assertAlmostEqual(
            (b ^ a)(dot),
            a(dot) * b(dot)
        )

        # (b ★ a)(/) = a(●)b(/) + a(/)b(●)²
        self.assertAlmostEqual(
            (b ^ a)(line),
            a(dot) * b(line) + a(line) * b(dot) ** 2
        )

        # (b ★ a)(Y) = a(●)b(Y) + 2a(/)b(●)b(/) + a(Y)b(●)³
        self.assertAlmostEqual(
            (b ^ a)(fork),
            a(dot) * b(fork) + 2 * a(line) * b(dot) * b(line) + a(fork) * b(dot) ** 3
        )


# ===========================================================================
# Reference tests verified against explicit tables in published papers
#
# [CEM2] D. Calaque, K. Ebrahimi-Fard, D. Manchon, "Two interacting Hopf
#        algebras of trees", Adv. Appl. Math. 47 (2011). (arxiv: 0806.2238)
# [M] D. Manchon, "Lois pre-Lie en interaction",
#     Comm. Algebra 39 (2011). (arxiv: 0811.2153)
# [CHV2] P. Chartier, E. Hairer, G. Vilmart, "Algebraic structures of
#        B-series", Found. Comput. Math. 10 (2010). (HAL: inria-00598369)
# ===========================================================================

# Tree definitions for reference tests — all 17 non-planar rooted trees
# through order 5

# Order 1
bullet   = T([])

# Order 2
chain2   = T([[]])

# Order 3
cherry   = T([[], []])
chain3   = T([[[]]])

# Order 4
corolla3 = T([[], [], []])
t43      = T([[], [[]]])
b_cherry = T([[[], []]])
chain4   = T([[[[]]]])

# Order 5
corolla4    = T([[], [], [], []])
bullets_c2  = T([[], [], [[]]])
bullet_ch   = T([[], [[], []]])
bullet_c3   = T([[], [[[]]]])
two_chain2s = T([[[]], [[]]])
b_corolla3  = T([[[], [], []]])
b_t43       = T([[[], [[]]]])
bb_cherry   = T([[[[], []]]])
chain5      = T([[[[[]]]]])

# Shorthand list_reprs for building expected dicts
B   = bullet.list_repr
C2  = chain2.list_repr
C3  = chain3.list_repr
CH  = cherry.list_repr
C4  = chain4.list_repr
T43 = t43.list_repr
CO3 = corolla3.list_repr
C5  = chain5.list_repr
CO4 = corolla4.list_repr


def _tps_dict(tps):
    """Convert TensorProductSum to {(left_key, right_key): coeff} dict."""
    d = {}
    for c, l, r in tps.term_list:
        lk = tuple(sorted((t.list_repr for t in l.tree_list
                           if t.list_repr is not None), key=str))
        rk = tuple(sorted((t.list_repr for t in r.tree_list
                           if t.list_repr is not None), key=str))
        key = (lk, rk)
        d[key] = d.get(key, 0) + c
    return {k: v for k, v in d.items() if v != 0}


# ---------------------------------------------------------------------------
# [CEM2] Lines 642-643: Explicit coproduct examples (Section 4.1)
# ---------------------------------------------------------------------------

class ExplicitCoproductTests(unittest.TestCase):
    """Coproduct examples from [CEM2] lines 642-643."""

    def _check(self, tree, expected):
        result = _tps_dict(coproduct_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"CEM coproduct({tree.list_repr})")

    def test_chain3(self):
        """Line 642: Delta(E_2) = E_2 x bullet + bullet x E_2 + 2*E_1 x E_1"""
        self._check(chain3, {
            ((C3,), (B,)): 1,
            ((B,), (C3,)): 1,
            ((C2,), (C2,)): 2,
        })

    def test_corolla3(self):
        """Line 643: Delta(C_3) = C_3 x bullet + bullet x C_3 + 3*C_2 x E_1 + 3*E_1 x C_2"""
        self._check(corolla3, {
            ((CO3,), (B,)): 1,
            ((B,), (CO3,)): 1,
            ((CH,), (C2,)): 3,
            ((C2,), (CH,)): 3,
        })


# ---------------------------------------------------------------------------
# [CEM2] Lines 764-771: Ladder coproducts (Section 5)
# ---------------------------------------------------------------------------

class LadderCoproductTests(unittest.TestCase):
    """Ladder coproduct values from [CEM2] lines 764-771."""

    def _check(self, tree, expected):
        result = _tps_dict(coproduct_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"CEM coproduct({tree.list_repr})")

    def test_bullet(self):
        self._check(bullet, {
            ((B,), (B,)): 1,
        })

    def test_chain2(self):
        self._check(chain2, {
            ((C2,), (B,)): 1,
            ((B,), (C2,)): 1,
        })

    def test_chain3(self):
        self._check(chain3, {
            ((C3,), (B,)): 1,
            ((B,), (C3,)): 1,
            ((C2,), (C2,)): 2,
        })

    def test_chain4(self):
        self._check(chain4, {
            ((C4,), (B,)): 1,
            ((B,), (C4,)): 1,
            ((C3,), (C2,)): 2,
            ((C2,), (C3,)): 3,
            ((C2, C2), (C2,)): 1,
        })

    def test_chain5(self):
        self._check(chain5, {
            ((C5,), (B,)): 1,
            ((B,), (C5,)): 1,
            ((C4,), (C2,)): 2,
            ((C3,), (C3,)): 3,
            ((C2,), (C4,)): 4,
            ((C2, C2), (C3,)): 3,
            ((C3, C2), (C2,)): 2,
        })


# ---------------------------------------------------------------------------
# [CEM2] Line 712: Corolla coproduct formula (Proposition 5.1)
# ---------------------------------------------------------------------------

class CorollaCoproductTests(unittest.TestCase):
    """Corolla coproduct from [CEM2] Proposition 5.1 (line 712)."""

    corollas = [bullet, chain2, cherry, corolla3, corolla4]
    corolla_reprs = [B, C2, CH, CO3, CO4]

    def _check(self, tree, expected):
        result = _tps_dict(coproduct_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"CEM coproduct({tree.list_repr})")

    def test_corolla_C2(self):
        expected = {}
        n = 2
        for p in range(n + 1):
            lk = (self.corolla_reprs[p],)
            rk = (self.corolla_reprs[n - p],)
            expected[(lk, rk)] = comb(n, p)
        self._check(cherry, expected)

    def test_corolla_C3(self):
        expected = {}
        n = 3
        for p in range(n + 1):
            lk = (self.corolla_reprs[p],)
            rk = (self.corolla_reprs[n - p],)
            expected[(lk, rk)] = comb(n, p)
        self._check(corolla3, expected)

    def test_corolla_C4(self):
        expected = {}
        n = 4
        for p in range(n + 1):
            lk = (self.corolla_reprs[p],)
            rk = (self.corolla_reprs[n - p],)
            expected[(lk, rk)] = comb(n, p)
        self._check(corolla4, expected)


# ---------------------------------------------------------------------------
# [M] Line 967: CEM coproduct of t43 from Manchon (2011)
# ---------------------------------------------------------------------------

class ManchonCoproductTests(unittest.TestCase):
    """CEM coproduct from [M] line 967."""

    def _check(self, tree, expected):
        result = _tps_dict(coproduct_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"CEM coproduct({tree.list_repr})")

    def test_t43_coproduct(self):
        self._check(t43, {
            ((B,),  (T43,)): 1,
            ((T43,), (B,)):  1,
            ((C2,), (CH,)):  2,
            ((C2,), (C3,)):  1,
            ((C3,), (C2,)):  1,
            ((CH,), (C2,)):  1,
            ((C2, C2), (C2,)): 1,
        })

    def test_t43_term_count(self):
        d = _tps_dict(coproduct_impl(t43))
        self.assertEqual(len(d), 7)
        self.assertEqual(sum(d.values()), 8)


# ---------------------------------------------------------------------------
# [CHV2] Page 12: Bushy tree (corolla) omega = Bernoulli numbers
# ---------------------------------------------------------------------------

class BushyTreeOmegaTests(unittest.TestCase):
    """Omega values for bushy trees from [CHV2] page 12."""

    def test_chain2_is_B1(self):
        result = Fraction(omega(chain2)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 2))

    def test_cherry_is_B2(self):
        result = Fraction(omega(cherry)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 6))

    def test_corolla3_is_B3(self):
        result = Fraction(omega(corolla3)).limit_denominator(1000)
        self.assertEqual(result, Fraction(0))

    def test_corolla4_is_B4(self):
        result = Fraction(omega(corolla4)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 30))


# ---------------------------------------------------------------------------
# [CHV2] Page 12: Tall tree (chain) omega = log(1+x) coefficients
# ---------------------------------------------------------------------------

class TallTreeOmegaTests(unittest.TestCase):
    """Omega values for tall trees from [CHV2] page 12."""

    def test_bullet(self):
        result = Fraction(omega(bullet)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1))

    def test_chain2(self):
        result = Fraction(omega(chain2)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 2))

    def test_chain3(self):
        result = Fraction(omega(chain3)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 3))

    def test_chain4(self):
        result = Fraction(omega(chain4)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 4))

    def test_chain5(self):
        result = Fraction(omega(chain5)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 5))


# ---------------------------------------------------------------------------
# [CEM2] Lines 1765-1779: Remaining omega values from complete table
# ---------------------------------------------------------------------------

class CEMOmegaOrder4Tests(unittest.TestCase):
    """Omega values for order-4 trees from [CEM2] lines 1765-1779."""

    def test_b_cherry(self):
        result = Fraction(omega(b_cherry)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 6))

    def test_t43(self):
        result = Fraction(omega(t43)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 12))


class CEMOmegaOrder5Tests(unittest.TestCase):
    """Omega values for order-5 trees from [CEM2] lines 1765-1779."""

    def test_bb_cherry(self):
        result = Fraction(omega(bb_cherry)).limit_denominator(1000)
        self.assertEqual(result, Fraction(3, 20))

    def test_b_t43(self):
        result = Fraction(omega(b_t43)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 10))

    def test_b_corolla3(self):
        result = Fraction(omega(b_corolla3)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 30))

    def test_bullet_chain3(self):
        result = Fraction(omega(bullet_c3)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 20))

    def test_two_chain2s(self):
        result = Fraction(omega(two_chain2s)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 30))

    def test_bullet_cherry(self):
        result = Fraction(omega(bullet_ch)).limit_denominator(1000)
        self.assertEqual(result, Fraction(1, 60))

    def test_bullets_chain2(self):
        result = Fraction(omega(bullets_c2)).limit_denominator(1000)
        self.assertEqual(result, Fraction(-1, 60))