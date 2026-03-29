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
from kauri import Tree, Map, ident, exact_weights, bck
from kauri import Tree as T
from kauri.bck.bck import coproduct_impl, antipode_impl

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

    # --- Literature-derived tests ---
    # [CHV] P. Chartier, E. Hairer, G. Vilmart,
    #       "Algebraic Structures of B-series",
    #       Found. Comput. Math. 10 (2010), pp. 407-427.
    #       https://doi.org/10.1007/s10208-010-9065-1
    # [F]   L. Foissy, "An introduction to Hopf algebras of trees",
    #       Preprint, Universite de Reims.
    #       https://www2.mathematik.hu-berlin.de/~kreimer/wp-content/uploads/Foissy.pdf

    def test_coproduct_order_4_trident(self):
        """Δ_CK for [[],[],[]] (root with 3 children).
        Reference: [F] §1.2, Definition p. 4.
        """
        expected = (
            T([[],[],[]]) @ T()
            + T() @ T([[],[],[]])
            + 3 * T([]) @ T([[],[]])
            + 3 * T([]) * T([]) @ T([[]])
            + T([]) * T([]) * T([]) @ T([])
        )
        self.assertEqual(expected, bck.coproduct(T([[],[],[]])))

    def test_coproduct_order_4_asymmetric(self):
        """Δ_CK for [[],[[]]] (root with leaf and 2-chain subtree).
        Reference: [F] §1.2.
        """
        expected = (
            T([[],[[]]]) @ T()
            + T() @ T([[],[[]]])
            + T([]) @ T([[[]]])
            + T([]) @ T([[],[]])
            + T([[]]) @ T([[]])
            + T([]) * T([[]]) @ T([])
            + T([]) * T([]) @ T([[]])
        )
        self.assertEqual(expected, bck.coproduct(T([[],[[]]])))

    def test_coproduct_order_4_cherry(self):
        """Δ_CK for [[[],[]]] (root -> child -> two grandchildren).
        Reference: [F] §1.2.
        """
        expected = (
            T([[[],[]]]) @ T()
            + T() @ T([[[],[]]])
            + T([[],[]]) @ T([])
            + 2 * T([]) @ T([[[]]])
            + T([]) * T([]) @ T([[]])
        )
        self.assertEqual(expected, bck.coproduct(T([[[],[]]]))  )

    def test_coproduct_order_4_tall_ladder(self):
        """Δ_CK for the 4-node ladder [[[[]]]].
        Reference: [F] §3.2, p. 20.
        """
        expected = (
            T([[[[]]]]) @ T()
            + T([[[]]]) @ T([])
            + T([[]]) @ T([[]])
            + T([]) @ T([[[]]])
            + T() @ T([[[[]]]])
        )
        self.assertEqual(expected, bck.coproduct(T([[[[]]]])))

    def test_ladder_coproduct(self):
        """Δ(l_n) = Σ_{i=0}^{n} l_i ⊗ l_{n-i} for ladder trees.
        Reference: [F] §3.2, p. 20.
        """
        ladders = [T(None), T([]), T([[]]), T([[[]]]), T([[[[]]]])]
        for n in range(1, len(ladders)):
            terms = [ladders[i] @ ladders[n - i] for i in range(n + 1)]
            expected = terms[0]
            for t in terms[1:]:
                expected = expected + t
            self.assertEqual(expected, bck.coproduct(ladders[n]),
                             msg=f"Ladder l_{n}")

    def test_antipode_foissy_order_4_trident(self):
        """S([[],[],[]]) = ●⁴ - 3●²/ + 3●Y - trident.
        Reference: [F] Theorem 2, p. 7.
        """
        expected = (
            T([]) ** 4
            - 3 * T([[]]) * T([]) * T([])
            + 3 * T([[],[]]) * T([])
            - T([[],[],[]])
        )
        self.assertEqual(expected, bck.antipode(T([[],[],[]])))

    def test_antipode_foissy_order_3(self):
        """S(Y) = -Y + 2●/ - ●³ and S(chain₃) = -chain₃ + 2●/ - ●³.
        Reference: [F] Theorem 2, p. 7.
        """
        self.assertEqual(
            -T([[],[]]) + 2 * T([[]]) * T([]) - T([]) ** 3,
            bck.antipode(T([[],[]]))
        )
        self.assertEqual(
            -T([[[]]]) + 2 * T([[]]) * T([]) - T([]) ** 3,
            bck.antipode(T([[[]]]))
        )


# ===========================================================================
# Reference tests verified against explicit tables in published papers
#
# [B] C. Brouder, "Runge-Kutta methods and renormalization",
#     Eur. Phys. J. C 12 (2000), 521-534. (arxiv: hep-th/9904014)
# [FGB] H. Figueroa, J.M. Gracia-Bondia, "On the antipode of Kreimer's
#       Hopf algebra", Eur. Phys. J. C (2001). (arxiv: hep-th/9912170)
# ===========================================================================

# Tree definitions for reference tests
bullet      = T([])
chain2      = T([[]])
chain3      = T([[[]]])
cherry      = T([[], []])
chain4      = T([[[[]]]])
b_cherry    = T([[[], []]])
t43         = T([[], [[]]])
corolla3    = T([[], [], []])
two_chain2s = T([[[]], [[]]])

# Canonical keys for dict comparison
B   = bullet.sorted_list_repr()
C2  = chain2.sorted_list_repr()
C3  = chain3.sorted_list_repr()
CH  = cherry.sorted_list_repr()
C4  = chain4.sorted_list_repr()
BC  = b_cherry.sorted_list_repr()
T43 = t43.sorted_list_repr()
CO3 = corolla3.sorted_list_repr()
T2C = two_chain2s.sorted_list_repr()


def _fk(*reprs):
    """Build a canonical forest key from tree reprs (sorted by str)."""
    return tuple(sorted(reprs, key=str))


def _tps_dict(tps):
    """Convert TensorProductSum to {(left_key, right_key): coeff} dict."""
    d = {}
    for c, l, r in tps.term_list:
        lk = _fk(*(t.sorted_list_repr() for t in l.tree_list
                    if t.list_repr is not None))
        rk = _fk(*(t.sorted_list_repr() for t in r.tree_list
                    if t.list_repr is not None))
        key = (lk, rk)
        d[key] = d.get(key, 0) + c
    return {k: v for k, v in d.items() if v != 0}


def _fs_dict(fs):
    """Convert ForestSum to {forest_key: coeff} dict."""
    d = {}
    for c, f in fs.term_list:
        key = _fk(*(t.sorted_list_repr() for t in f.tree_list
                     if t.list_repr is not None))
        d[key] = d.get(key, 0) + c
    return {k: v for k, v in d.items() if v != 0}


class BrouderCoproductTests(unittest.TestCase):
    """[B] BCK coproduct, Appendix lines 1703-1779."""

    def _check(self, tree, expected):
        result = _tps_dict(coproduct_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"BCK coproduct({tree.sorted_list_repr()})")

    def test_bullet(self):
        """Line 1705: Delta(bullet) = bullet x 1 + 1 x bullet"""
        self._check(bullet, {
            (_fk(B), ()): 1,
            ((), _fk(B)): 1,
        })

    def test_chain2(self):
        """Line 1709: Delta(t2) = t2 x 1 + 1 x t2 + bullet x bullet"""
        self._check(chain2, {
            (_fk(C2), ()): 1,
            ((), _fk(C2)): 1,
            (_fk(B), _fk(B)): 1,
        })

    def test_chain3(self):
        """Line 1715: Delta(t31) = t31 x 1 + 1 x t31 + t2 x bullet + bullet x t2"""
        self._check(chain3, {
            (_fk(C3), ()): 1,
            ((), _fk(C3)): 1,
            (_fk(C2), _fk(B)): 1,
            (_fk(B), _fk(C2)): 1,
        })

    def test_cherry(self):
        """Line 1724: Delta(t32) = t32 x 1 + 1 x t32 + bb x bullet + 2*bullet x t2"""
        self._check(cherry, {
            (_fk(CH), ()): 1,
            ((), _fk(CH)): 1,
            (_fk(B, B), _fk(B)): 1,
            (_fk(B), _fk(C2)): 2,
        })

    def test_chain4(self):
        """Line 1733: Delta(t41)"""
        self._check(chain4, {
            (_fk(C4), ()): 1,
            ((), _fk(C4)): 1,
            (_fk(B), _fk(C3)): 1,
            (_fk(C3), _fk(B)): 1,
            (_fk(C2), _fk(C2)): 1,
        })

    def test_b_cherry(self):
        """Line 1744: Delta(t42)"""
        self._check(b_cherry, {
            (_fk(BC), ()): 1,
            ((), _fk(BC)): 1,
            (_fk(B), _fk(C3)): 2,
            (_fk(CH), _fk(B)): 1,
            (_fk(B, B), _fk(C2)): 1,
        })

    def test_t43(self):
        """Line 1755: Delta(t43)"""
        self._check(t43, {
            (_fk(T43), ()): 1,
            ((), _fk(T43)): 1,
            (_fk(B), _fk(C3)): 1,
            (_fk(B), _fk(CH)): 1,
            (_fk(C2), _fk(C2)): 1,
            (_fk(B, B), _fk(C2)): 1,
            (_fk(B, C2), _fk(B)): 1,
        })

    def test_corolla3(self):
        """Line 1769: Delta(t44)"""
        self._check(corolla3, {
            (_fk(CO3), ()): 1,
            ((), _fk(CO3)): 1,
            (_fk(B), _fk(CH)): 3,
            (_fk(B, B), _fk(C2)): 3,
            (_fk(B, B, B), _fk(B)): 1,
        })


class BrouderAntipodeTests(unittest.TestCase):
    """[B] BCK antipode, Appendix lines 1783-1818."""

    def _check(self, tree, expected):
        result = _fs_dict(antipode_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"BCK S({tree.sorted_list_repr()})")

    def test_bullet(self):
        self._check(bullet, {_fk(B): -1})

    def test_chain2(self):
        self._check(chain2, {
            _fk(C2): -1,
            _fk(B, B): 1,
        })

    def test_chain3(self):
        self._check(chain3, {
            _fk(C3): -1,
            _fk(B, C2): 2,
            _fk(B, B, B): -1,
        })

    def test_cherry(self):
        self._check(cherry, {
            _fk(CH): -1,
            _fk(B, C2): 2,
            _fk(B, B, B): -1,
        })

    def test_chain4(self):
        self._check(chain4, {
            _fk(C4): -1,
            _fk(B, C3): 2,
            _fk(C2, C2): 1,
            _fk(B, B, C2): -3,
            _fk(B, B, B, B): 1,
        })

    def test_b_cherry(self):
        self._check(b_cherry, {
            _fk(BC): -1,
            _fk(B, C3): 2,
            _fk(B, CH): 1,
            _fk(B, B, C2): -3,
            _fk(B, B, B, B): 1,
        })

    def test_t43(self):
        self._check(t43, {
            _fk(T43): -1,
            _fk(B, C3): 1,
            _fk(B, CH): 1,
            _fk(C2, C2): 1,
            _fk(B, B, C2): -3,
            _fk(B, B, B, B): 1,
        })

    def test_corolla3(self):
        self._check(corolla3, {
            _fk(CO3): -1,
            _fk(B, CH): 3,
            _fk(B, B, C2): -3,
            _fk(B, B, B, B): 1,
        })


class FigueroaAntipodeOrder4Tests(unittest.TestCase):
    """[FGB] BCK antipode cross-check, eq. 2.7 (lines 540-546)."""

    def test_t43_antipode(self):
        result = _fs_dict(antipode_impl(t43))
        expected = {
            _fk(T43): -1,
            _fk(B, CH): 1,
            _fk(C2, C2): 1,
            _fk(B, C3): 1,
            _fk(B, B, C2): -3,
            _fk(B, B, B, B): 1,
        }
        self.assertEqual(result, expected,
                         msg="BCK S(t43) cross-check with FGB eq. 2.7")


class FigueroaAntipodeOrder5Tests(unittest.TestCase):
    """[FGB] BCK antipode for order-5 tree, lines 551-559."""

    def test_two_chain2s_antipode(self):
        result = _fs_dict(antipode_impl(two_chain2s))
        expected = {
            _fk(T2C): -1,
            _fk(B, T43): 2,
            _fk(C2, C3): 2,
            _fk(B, B, CH): -1,
            _fk(B, C2, C2): -3,
            _fk(B, B, C3): -2,
            _fk(B, B, B, C2): 4,
            _fk(B, B, B, B, B): -1,
        }
        self.assertEqual(result, expected,
                         msg="BCK S(two_chain2s) from FGB lines 551-559")

    def test_two_chain2s_term_count(self):
        d = _fs_dict(antipode_impl(two_chain2s))
        self.assertEqual(len(d), 8)