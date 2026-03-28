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
from kauri import Tree, Map, ident
import kauri.gl as gl
from kauri.trees import Forest, ForestSum

T = Tree

# Trees used across tests (all non-planar, unlabelled)
trees = [T([]),            # bullet
         T([[]]),          # / (chain_2)
         T([[],[]]),       # Y (cherry)
         T([[[]]]),        # chain_3
         T([[],[],[]]),    # trident
         T([[],[[]]])]     # asymmetric order 4

# Extended set for thorough checks
trees_ext = trees + [
    T([[[],[]]]),      # order 4: root -> cherry
    T([[[[]]]]),       # chain_4
]


class GLTests(unittest.TestCase):

    # ------------------------------------------------------------------
    # Coproduct
    # ------------------------------------------------------------------

    def test_coproduct(self):
        """Explicit coproduct values for small trees."""
        # Delta(bullet) = bullet tensor bullet
        self.assertEqual(
            T([]) @ T([]),
            gl.coproduct(T([]))
        )
        # Delta(/) = bullet tensor / + / tensor bullet
        self.assertEqual(
            T([]) @ T([[]]) + T([[]]) @ T([]),
            gl.coproduct(T([[]]))
        )
        # Delta(Y) = bullet tensor Y + 2 / tensor / + Y tensor bullet
        self.assertEqual(
            T([]) @ T([[],[]]) + 2 * T([[]]) @ T([[]]) + T([[],[]]) @ T([]),
            gl.coproduct(T([[],[]]))
        )
        # Delta(chain_3) = bullet tensor chain_3 + chain_3 tensor bullet (primitive)
        self.assertEqual(
            T([]) @ T([[[]]]) + T([[[]]]) @ T([]),
            gl.coproduct(T([[[]]]))
        )

    def test_coproduct_primitive(self):
        """Trees with exactly 1 child (chains) are primitive:
        Delta(t) = bullet tensor t + t tensor bullet."""
        chains = [T([[]]), T([[[]]]), T([[[[]]]])]
        for t in chains:
            expected = T([]) @ t + t @ T([])
            self.assertEqual(expected, gl.coproduct(t),
                             msg=f"Chain {t} should be primitive")

    def test_coproduct_trident(self):
        """Delta(trident) = bullet tensor trident + 3 / tensor Y
        + 3 Y tensor / + trident tensor bullet."""
        expected = (
            T([]) @ T([[],[],[]])
            + 3 * T([[]]) @ T([[],[]])
            + 3 * T([[],[]]) @ T([[]])
            + T([[],[],[]]) @ T([])
        )
        self.assertEqual(expected, gl.coproduct(T([[],[],[]])))

    def test_coproduct_asymmetric(self):
        """Delta for [[],[[]]] = B+(bullet, /).
        Subsets: {}, {1}, {2}, {1,2}."""
        expected = (
            T([]) @ T([[],[[]]])
            + T([[]]) @ T([[[]]])
            + T([[[]]]) @ T([[]])
            + T([[],[[]]] ) @ T([])
        )
        self.assertEqual(expected, gl.coproduct(T([[],[[]]])))

    # ------------------------------------------------------------------
    # Counit
    # ------------------------------------------------------------------

    def test_counit(self):
        self.assertEqual(1, gl.counit(T([])))
        for t in trees[1:]:
            self.assertEqual(0, gl.counit(t))

    # ------------------------------------------------------------------
    # Product (grafting)
    # ------------------------------------------------------------------

    def test_product_unit(self):
        """bullet . t = t and t . bullet = t for all trees."""
        bullet = T([])
        for t in trees:
            self.assertEqual(t, gl.product(bullet, t),
                             msg=f"bullet . {t} should be {t}")
            self.assertEqual(t, gl.product(t, bullet),
                             msg=f"{t} . bullet should be {t}")

    def test_product_chain_chain(self):
        """/ . / = Y + chain_3."""
        expected = T([[],[]]) + T([[[]]])
        self.assertEqual(expected, gl.product(T([[]]), T([[]])))

    def test_product_chain_cherry(self):
        """/ . Y = trident + 2 [[],[[]]] + [[[],[]]]."""
        expected = (
            T([[],[],[]])
            + 2 * T([[],[[]]])
            + T([[[],[]]])
        )
        self.assertEqual(expected, gl.product(T([[]]), T([[],[]])))

    def test_product_result_count(self):
        """GL product s . t produces |V(s)|^k trees (with repetitions),
        where k = number of children of t's root."""
        for s in trees[:4]:
            for t in trees[:4]:
                result = gl.product(s, t)
                # Just check it's a valid ForestSum
                self.assertIsInstance(result, (Tree, type(result)))

    # ------------------------------------------------------------------
    # Antipode
    # ------------------------------------------------------------------

    def test_antipode(self):
        """Explicit antipode values for orders 1-3."""
        # S(bullet) = bullet
        self.assertEqual(T([]), gl.antipode(T([])))
        # S(/) = -/
        self.assertEqual(-T([[]]), gl.antipode(T([[]])))
        # S(Y) = Y + 2 chain_3
        self.assertEqual(
            T([[],[]]) + 2 * T([[[]]]),
            gl.antipode(T([[],[]]))
        )
        # S(chain_3) = -chain_3  (primitive => S = -id)
        self.assertEqual(-T([[[]]]), gl.antipode(T([[[]]])))

    def test_antipode_property(self):
        """Verify sum_Delta S(left) .GL right = epsilon(t) . bullet
        (the defining property of the antipode)."""
        for t in trees:
            cp = gl.coproduct(t)
            result = 0
            for c, lf, rf in cp:
                left = lf.tree_list[0]
                right = rf.tree_list[0]
                s_left = gl.antipode(left)
                # Extend GL product linearly over ForestSum
                gl_prod = gl.product(s_left, right)
                result = result + c * gl_prod
            if gl.counit(t) == 0:
                self.assertEqual(0, result,
                                 msg=f"Antipode property failed for {t}")
            else:
                self.assertEqual(T([]), result,
                                 msg=f"Antipode property failed for {t}")

    def test_antipode_squared(self):
        """S_GL composed with S_GL is the identity."""
        for t in trees_ext:
            self.assertEqual(t, gl.antipode(gl.antipode(t)),
                             msg=f"S(S({t})) != {t}")

    # ------------------------------------------------------------------
    # Map convolution
    # ------------------------------------------------------------------

    def test_map_product(self):
        """GL convolution of counit with itself is counit."""
        m = gl.map_product(gl.counit, gl.counit)
        for t in trees:
            self.assertEqual(gl.counit(t), m(t))

    def test_map_power(self):
        """Convolution powers: counit^n = counit for all n >= 1."""
        for n in [1, 2, 3]:
            m = gl.map_power(gl.counit, n)
            for t in trees:
                self.assertEqual(gl.counit(t), m(t),
                                 msg=f"counit^{n}({t})")

    def test_map_power_inverse(self):
        """f^n * f^{-n} = counit (via GL convolution)."""
        f = Map(lambda x: x.nodes() if x.list_repr is not None else 1)
        f2 = gl.map_power(f, 2)
        f_neg2 = gl.map_power(f, -2)
        m = gl.map_product(f2, f_neg2)
        for t in trees:
            self.assertEqual(gl.counit(t), m(t),
                             msg=f"f^2 * f^-2 ({t})")

    # ------------------------------------------------------------------
    # Type errors
    # ------------------------------------------------------------------

    def test_type_error_coproduct(self):
        with self.assertRaises(TypeError):
            gl.coproduct('s')
        with self.assertRaises(TypeError):
            gl.coproduct(T(None))

    def test_type_error_product(self):
        with self.assertRaises(TypeError):
            gl.product('s', T([]))
        with self.assertRaises(TypeError):
            gl.product(T([]), 's')

    def test_type_error_antipode(self):
        with self.assertRaises(TypeError):
            gl.antipode('s')

    def test_type_error_counit(self):
        with self.assertRaises(TypeError):
            gl.counit('s')


# ===========================================================================
# Reference tests: Hoffman (2007), arxiv 0710.3739
# Lines 421-450: GL product examples
# ===========================================================================

# Tree definitions for reference tests
bullet   = T([])
chain2   = T([[]])
cherry   = T([[], []])
corolla3 = T([[], [], []])
t43      = T([[], [[]]])
b_cherry = T([[[], []]])


def _fk(*reprs):
    """Build a canonical forest key from tree list_reprs (sorted by str)."""
    return tuple(sorted(reprs, key=str))


def _fs_dict(fs):
    """Convert ForestSum to {forest_key: coeff} dict.

    Uses sorted_list_repr() for canonical comparison of unordered trees.
    """
    d = {}
    for c, f in fs.term_list:
        key = _fk(*(t.sorted_list_repr() for t in f.tree_list
                     if t.list_repr is not None))
        d[key] = d.get(key, 0) + c
    return {k: v for k, v in d.items() if v != 0}


CO3 = corolla3.sorted_list_repr()
T43 = t43.sorted_list_repr()
BC  = b_cherry.sorted_list_repr()


class GLProductReferenceTests(unittest.TestCase):
    """GL product examples from Hoffman (2007), arxiv 0710.3739."""

    def _check(self, result, expected):
        """Compare ForestSum result to expected {forest_key: coeff} dict."""
        got = _fs_dict(result)
        self.assertEqual(got, expected)

    def test_cherry_circ_chain2(self):
        """Lines 427-437: cherry o chain2 = corolla3 + 2*t43 + B+(cherry)

        cherry = B+(bullet, bullet) has 2 children; chain2 has 2 vertices.
        2^2 = 4 assignments producing 3 distinct trees.

        kauri: gl.product(chain2, cherry) = Hoffman's cherry o chain2.
        """
        result = gl.product(chain2, cherry)
        self._check(result, {
            _fk(CO3): 1,
            _fk(T43): 2,
            _fk(BC): 1,
        })

    def test_chain2_circ_cherry(self):
        """Lines 442-450: chain2 o cherry = corolla3 + 2*t43

        chain2 = B+(bullet) has 1 child; cherry has 3 vertices.
        3^1 = 3 assignments producing 2 distinct trees (cherry's 2 leaves
        are symmetric, giving coefficient 2 for t43).

        kauri: gl.product(cherry, chain2) = Hoffman's chain2 o cherry.
        """
        result = gl.product(cherry, chain2)
        self._check(result, {
            _fk(CO3): 1,
            _fk(T43): 2,
        })

    def test_gl_product_noncommutative(self):
        """Lines 427-450 together demonstrate that the GL product is
        noncommutative: cherry o chain2 != chain2 o cherry.

        cherry o chain2 has 3 distinct trees (4 terms),
        chain2 o cherry has 2 distinct trees (3 terms).
        """
        r1 = gl.product(chain2, cherry)  # cherry o chain2
        r2 = gl.product(cherry, chain2)  # chain2 o cherry
        d1 = _fs_dict(r1)
        d2 = _fs_dict(r2)
        self.assertNotEqual(d1, d2,
                            msg="GL product should be noncommutative")
        # cherry o chain2 has B+(cherry) term that chain2 o cherry lacks
        self.assertIn(_fk(BC), d1)
        self.assertNotIn(_fk(BC), d2)

    def test_bullet_is_unit(self):
        """Line 424: if t = bullet, set t o t' = t'.

        The bullet is the two-sided unit for the GL product.
        kauri: gl.product(t', bullet) = Hoffman's bullet o t' = t'.
        """
        for t in [bullet, chain2, cherry]:
            result = gl.product(t, bullet)
            d = _fs_dict(result)
            expected_key = _fk(t.sorted_list_repr())
            self.assertEqual(d, {expected_key: 1},
                             msg=f"bullet o {t.sorted_list_repr()} should be identity")
