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

"""
Tests for the NCK (noncommutative Connes-Kreimer) Hopf algebra (kauri.nck).

References:
    - H. Munthe-Kaas, W. Wright, "On the Hopf algebraic structure of Lie
      group integrators", Found. Comput. Math. 8 (2008), 227-257.
    - A. Connes, D. Kreimer, "Hopf algebras, renormalization and
      noncommutative geometry", Comm. Math. Phys. 199 (1998), 203-242.
"""

import unittest
from kauri.trees import PlanarTree, NoncommutativeForest, OrderedForest, ForestSum, EMPTY_PLANAR_TREE, EMPTY_ORDERED_FOREST
from kauri.maps import Map
import kauri.nck as nck
from kauri.nck.nck import _forest_sum_mul_tree
from kauri.generic_algebra import anti_forest_apply

PT = PlanarTree

# Test trees: empty tree + all planar trees up to order 4
trees = [PT(None), PT([]), PT([[]]), PT([[],[]]), PT([[[]]]),
         PT([[],[],[]]), PT([[],[[]]]), PT([[[]],[]])]

# Trees without the empty tree (non-unit elements)
nonunit_trees = trees[1:]


def _as_fs(t):
    """Wrap a PlanarTree as a ForestSum for equality comparison."""
    if t.list_repr is None:
        return ForestSum(((1, EMPTY_ORDERED_FOREST),))
    return ForestSum(((1, t.as_ordered_forest()),))


class TestCounit(unittest.TestCase):

    def test_counit_empty(self):
        self.assertEqual(1, nck.counit(PT(None)))

    def test_counit_nonunit(self):
        for t in nonunit_trees:
            self.assertEqual(0, nck.counit(t),
                             msg=f"counit({t.list_repr}) should be 0")


class TestCoproduct(unittest.TestCase):

    def test_coproduct_empty(self):
        cp = nck.coproduct(PT(None))
        self.assertEqual(len(cp), 1)
        c, left, right = cp[0]
        self.assertEqual(c, 1)
        self.assertEqual(left, EMPTY_ORDERED_FOREST)
        self.assertEqual(right, EMPTY_ORDERED_FOREST)

    def test_coproduct_bullet(self):
        cp = nck.coproduct(PT([]))
        # Delta(bullet) = empty tensor bullet + bullet tensor empty
        self.assertEqual(len(cp), 2)
        terms = {(left, right): c for c, left, right in cp}
        self.assertEqual(terms[(EMPTY_ORDERED_FOREST, PT([]).as_ordered_forest())], 1)
        self.assertEqual(terms[(PT([]).as_ordered_forest(), EMPTY_ORDERED_FOREST)], 1)

    def test_coproduct_chain2(self):
        cp = nck.coproduct(PT([[]]))
        # Delta(/) = / tensor empty + empty tensor / + bullet tensor bullet
        terms = {}
        for c, left, right in cp:
            terms[(left, right)] = terms.get((left, right), 0) + c
        self.assertEqual(terms[(OrderedForest((PT([[]]),)), EMPTY_ORDERED_FOREST)], 1)
        self.assertEqual(terms[(EMPTY_ORDERED_FOREST, OrderedForest((PT([[]]),)))], 1)
        self.assertEqual(terms[(OrderedForest((PT([]),)), OrderedForest((PT([]),)))], 1)
        self.assertEqual(len(terms), 3)

    def test_coproduct_cherry(self):
        cp = nck.coproduct(PT([[],[]]))
        terms = {}
        for c, left, right in cp:
            terms[(left, right)] = terms.get((left, right), 0) + c
        # Delta(Y) = Y tensor empty + empty tensor Y
        #           + 2 * bullet tensor chain2 + bullet*bullet tensor bullet
        self.assertEqual(terms[(OrderedForest((PT([[],[]]),)), EMPTY_ORDERED_FOREST)], 1)
        self.assertEqual(terms[(EMPTY_ORDERED_FOREST, OrderedForest((PT([[],[]]),)))] , 1)
        self.assertEqual(terms[(OrderedForest((PT([]),)), OrderedForest((PT([[]]),)))], 2)
        self.assertEqual(terms[(OrderedForest((PT([]), PT([]))), OrderedForest((PT([]),)))], 1)
        self.assertEqual(len(terms), 4)

    def test_coproduct_bullet_primitive(self):
        """Only the single-node tree is primitive in BCK: Delta(t) = empty tensor t + t tensor empty."""
        cp = nck.coproduct(PT([]))
        self.assertEqual(len(cp), 2)

    def test_coproduct_planar_sensitivity(self):
        """Different planar orderings give different coproducts."""
        t1 = PT([[[]],[]])   # B+(chain2, bullet)
        t2 = PT([[],[[]]])   # B+(bullet, chain2)
        cp1 = nck.coproduct(t1)
        cp2 = nck.coproduct(t2)
        # Convert to comparable sets
        terms1 = {(left, right): c for c, left, right in cp1}
        terms2 = {(left, right): c for c, left, right in cp2}
        self.assertNotEqual(terms1, terms2,
                            msg="Coproducts of [[[]],[]] and [[],[[]]] should differ")

    def test_type_error(self):
        with self.assertRaises(TypeError):
            nck.coproduct('s')
        with self.assertRaises(TypeError):
            nck.coproduct(42)


class TestAntipode(unittest.TestCase):

    def test_antipode_empty(self):
        """S(empty) = empty."""
        self.assertEqual(_as_fs(PT(None)), nck.antipode(PT(None)))

    def test_antipode_bullet(self):
        """S(bullet) = -bullet."""
        expected = ForestSum(((-1, PT([]).as_ordered_forest()),))
        self.assertEqual(expected, nck.antipode(PT([])))

    def test_antipode_chain2(self):
        """S(/) = -/ + bullet*bullet."""
        expected = (
            ForestSum(((-1, OrderedForest((PT([[]]),))),))
            + ForestSum(((1, OrderedForest((PT([]), PT([])))),))
        )
        self.assertEqual(expected, nck.antipode(PT([[]])))

    def test_antipode_cherry(self):
        """S(Y) = -Y + 2*bullet*/ - bullet*bullet*bullet."""
        expected = (
            ForestSum(((-1, OrderedForest((PT([[],[]]),))),))
            + ForestSum(((2, OrderedForest((PT([]), PT([[]])))),))
            + ForestSum(((-1, OrderedForest((PT([]), PT([]), PT([])))),))
        )
        self.assertEqual(expected, nck.antipode(PT([[],[]])))

    def test_antipode_planar_sensitivity(self):
        """Different planar orderings produce different antipodes."""
        s1 = nck.antipode(PT([[[]],[]]))
        s2 = nck.antipode(PT([[],[[]]]))
        self.assertNotEqual(s1, s2)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            nck.antipode('s')


class TestAntipodeProperty(unittest.TestCase):
    """Verify m(S tensor id) Delta = eta o epsilon (left antipode property)."""

    def test_left_antipode(self):
        for t in nonunit_trees:
            cp = nck.coproduct(t)
            result = 0
            for c, left_forest, right_forest in cp:
                right_tree = right_forest[0]
                # S is an anti-homomorphism: S(t1*t2) = S(t2)*S(t1)
                s_left = anti_forest_apply(left_forest, nck.antipode.func)
                if right_tree.list_repr is not None:
                    term = _forest_sum_mul_tree(s_left, right_tree)
                else:
                    term = s_left
                result = result + c * term
            self.assertEqual(0, result,
                             msg=f"Left antipode property failed for {t.list_repr}")

    def test_right_antipode(self):
        """Verify m(id tensor S) Delta = eta o epsilon."""
        for t in nonunit_trees:
            cp = nck.coproduct(t)
            result = 0
            for c, left_forest, right_forest in cp:
                right_tree = right_forest[0]
                s_right = nck.antipode(right_tree)
                # Multiply left_forest * S(right)
                # left_forest is NoncommutativeForest, s_right is ForestSum
                # NCF * ForestSum works via NoncommutativeForest.__mul__
                term = left_forest * s_right
                result = result + c * term
            self.assertEqual(0, result,
                             msg=f"Right antipode property failed for {t.list_repr}")


class TestCounitOfAntipode(unittest.TestCase):
    """Verify epsilon o S = epsilon."""

    def test_counit_of_antipode(self):
        for t in trees:
            s_t = nck.antipode(t)
            # Apply counit linearly to S(t)
            eps_s = 0
            for c, forest in s_t.term_list:
                val = 1
                for tree in forest.tree_list:
                    val *= nck.counit(tree)
                eps_s += c * val
            self.assertEqual(nck.counit(t), eps_s,
                             msg=f"eps(S({t.list_repr})) != eps({t.list_repr})")


class TestAntipodeInvolution(unittest.TestCase):
    """S^2 = id holds for commutative or cocommutative Hopf algebras.
    The planar BCK is neither, so S^2 != id in general."""

    def test_involution_single_child_trees(self):
        """S^2 = id for trees where S(t) contains only single-tree forests."""
        for t in [PT([]), PT([[]]), PT([[[]]])]:
            s2 = nck.antipode(nck.antipode(t))
            self.assertEqual(_as_fs(t), s2,
                             msg=f"S^2({t.list_repr}) should equal t for this tree")

    def test_not_involution(self):
        """S^2 != id for trees whose antipode involves multi-tree forests."""
        for t in [PT([[],[]]), PT([[[]],[]]), PT([[],[[]]])]:
            s2 = nck.antipode(nck.antipode(t))
            self.assertNotEqual(_as_fs(t), s2,
                                msg=f"S^2 should NOT be id for {t.list_repr}")


class TestConvolution(unittest.TestCase):

    def test_counit_conv_identity(self):
        """f * epsilon = f = epsilon * f."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        f_eps = nck.map_product(f, nck.counit)
        eps_f = nck.map_product(nck.counit, f)
        for t in trees:
            self.assertEqual(f(t), f_eps(t),
                             msg=f"(f*eps)({t.list_repr}) != f({t.list_repr})")
            self.assertEqual(f(t), eps_f(t),
                             msg=f"(eps*f)({t.list_repr}) != f({t.list_repr})")

    def test_conv_associativity(self):
        """(f * g) * h = f * (g * h)."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        g = Map(lambda t: (-1)**(t.nodes()) if t.list_repr is not None else 1)
        h = Map(lambda t: t.nodes()**2 if t.list_repr is not None else 1)
        fg_h = nck.map_product(nck.map_product(f, g), h)
        f_gh = nck.map_product(f, nck.map_product(g, h))
        for t in trees:
            self.assertEqual(fg_h(t), f_gh(t),
                             msg=f"(f*g)*h != f*(g*h) at {t.list_repr}")

    def test_map_product_counit(self):
        """counit * counit = counit."""
        m = nck.map_product(nck.counit, nck.counit)
        for t in trees:
            self.assertEqual(nck.counit(t), m(t))


class TestMapPower(unittest.TestCase):

    def test_map_power_counit(self):
        """counit^n = counit for all n >= 1."""
        for n in [1, 2, 3]:
            m = nck.map_power(nck.counit, n)
            for t in trees:
                self.assertEqual(nck.counit(t), m(t),
                                 msg=f"counit^{n}({t.list_repr})")

    def test_map_power_zero(self):
        """f^0 = counit."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        m = nck.map_power(f, 0)
        for t in trees:
            self.assertEqual(nck.counit(t), m(t),
                             msg=f"f^0({t.list_repr}) should be counit")

    def test_map_power_inverse(self):
        """f^n * f^{-n} = counit."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        f2 = nck.map_power(f, 2)
        f_neg2 = nck.map_power(f, -2)
        m = nck.map_product(f2, f_neg2)
        for t in trees:
            self.assertEqual(nck.counit(t), m(t),
                             msg=f"f^2 * f^-2 ({t.list_repr})")

    def test_type_error_map_product(self):
        with self.assertRaises(TypeError):
            nck.map_product('a', nck.counit)
        with self.assertRaises(TypeError):
            nck.map_product(nck.counit, 'b')

    def test_type_error_map_power(self):
        with self.assertRaises(TypeError):
            nck.map_power('a', 2)
        with self.assertRaises(TypeError):
            nck.map_power(nck.counit, 2.5)
