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

"""
Tests for the planar Grossman-Larson Hopf algebra (kauri.pgl).
"""

import unittest
from kauri.trees import PlanarTree, NoncommutativeForest, OrderedForest, ForestSum, EMPTY_ORDERED_FOREST
from kauri.maps import Map
import kauri.pgl as pgl

PT = PlanarTree

# Test trees: all planar trees up to order 4
trees = [PT([]),            # bullet
         PT([[]]),          # chain_2
         PT([[],[]]),       # cherry
         PT([[[]]]),        # chain_3
         PT([[],[],[]]),    # trident
         PT([[],[[]]]),     # B+(bullet, chain_2)
         PT([[[]],[]])]     # B+(chain_2, bullet)

nonunit_trees = trees[1:]


def _as_fs(t):
    """Wrap a PlanarTree as a ForestSum for equality comparison."""
    return ForestSum(((1, t.as_ordered_forest()),))


class TestCounit(unittest.TestCase):

    def test_counit_bullet(self):
        self.assertEqual(1, pgl.counit(PT([])))

    def test_counit_nonunit(self):
        for t in nonunit_trees:
            self.assertEqual(0, pgl.counit(t),
                             msg=f"counit({t.list_repr}) should be 0")


class TestCoproduct(unittest.TestCase):

    def test_coproduct_bullet(self):
        """Delta(bullet) = bullet tensor bullet (1 term)."""
        cp = pgl.coproduct(PT([]))
        self.assertEqual(len(cp), 1)
        c, left, right = cp[0]
        self.assertEqual(c, 1)
        self.assertEqual(left, OrderedForest((PT([]),)))
        self.assertEqual(right, OrderedForest((PT([]),)))

    def test_coproduct_chain2(self):
        """Chain trees are primitive: Delta = bullet tensor t + t tensor bullet."""
        cp = pgl.coproduct(PT([[]]))
        terms = {(left, right): c for c, left, right in cp}
        self.assertEqual(len(terms), 2)
        self.assertEqual(terms[(OrderedForest((PT([]),)), OrderedForest((PT([[]]),)))], 1)
        self.assertEqual(terms[(OrderedForest((PT([[]]),)), OrderedForest((PT([]),)))], 1)

    def test_coproduct_cherry(self):
        """Delta(Y) = bullet tensor Y + 2 chain_2 tensor chain_2 + Y tensor bullet."""
        cp = pgl.coproduct(PT([[],[]]))
        terms = {}
        for c, left, right in cp:
            terms[(left, right)] = terms.get((left, right), 0) + c
        # bullet tensor cherry
        self.assertEqual(terms[(OrderedForest((PT([]),)), OrderedForest((PT([[],[]]),)))] , 1)
        # cherry tensor bullet
        self.assertEqual(terms[(OrderedForest((PT([[],[]]),)), OrderedForest((PT([]),)))], 1)
        # chain_2 tensor chain_2 (both children are identical bullets, so coeff=2)
        self.assertEqual(terms[(OrderedForest((PT([[]]),)), OrderedForest((PT([[]]),)))], 2)
        self.assertEqual(len(terms), 3)

    def test_coproduct_primitive(self):
        """Trees with 1 child (chains) are primitive."""
        chains = [PT([[]]), PT([[[]]]), PT([[[[]]]])]
        for t in chains:
            cp = pgl.coproduct(t)
            terms = {(left, right): c for c, left, right in cp}
            self.assertEqual(len(terms), 2, msg=f"Chain {t.list_repr} should be primitive")
            self.assertIn((OrderedForest((PT([]),)), OrderedForest((t,))), terms)
            self.assertIn((OrderedForest((t,)), OrderedForest((PT([]),))), terms)

    def test_coproduct_planar_sensitivity(self):
        """Different planar orderings give different coproducts."""
        cp1 = pgl.coproduct(PT([[],[[]]]))
        cp2 = pgl.coproduct(PT([[[]],[]]))
        terms1 = {(l, r): c for c, l, r in cp1}
        terms2 = {(l, r): c for c, l, r in cp2}
        self.assertNotEqual(terms1, terms2)

    def test_coproduct_asymmetric(self):
        """Delta for B+(bullet, chain_2): 4 terms from 2 children."""
        cp = pgl.coproduct(PT([[],[[]]]))
        terms = {(left, right): c for c, left, right in cp}
        # {} -> bullet tensor B+(bullet, chain_2)
        self.assertEqual(terms[(OrderedForest((PT([]),)), OrderedForest((PT([[],[[]]]),))) ], 1)
        # {1} -> chain_2 tensor B+(chain_2)=chain_3
        self.assertEqual(terms[(OrderedForest((PT([[]]),)), OrderedForest((PT([[[]]]),))) ], 1)
        # {2} -> chain_3 tensor chain_2
        self.assertEqual(terms[(OrderedForest((PT([[[]]]),)), OrderedForest((PT([[]]),)) )], 1)
        # {1,2} -> B+(bullet, chain_2) tensor bullet
        self.assertEqual(terms[(OrderedForest((PT([[],[[]]]),)), OrderedForest((PT([]),)) )], 1)
        self.assertEqual(len(terms), 4)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            pgl.coproduct('s')
        with self.assertRaises(TypeError):
            pgl.coproduct(PT(None))


class TestProduct(unittest.TestCase):

    def test_product_unit_left(self):
        """bullet . t = t for all trees."""
        bullet = PT([])
        for t in trees:
            result = pgl.product(bullet, t)
            self.assertEqual(_as_fs(t), result,
                             msg=f"bullet . {t.list_repr} should be {t.list_repr}")

    def test_product_unit_right(self):
        """t . bullet = t for all trees."""
        bullet = PT([])
        for t in trees:
            result = pgl.product(t, bullet)
            self.assertEqual(_as_fs(t), result,
                             msg=f"{t.list_repr} . bullet should be {t.list_repr}")

    def test_product_chain_chain(self):
        """chain_2 . chain_2 = cherry + chain_3 in planar GL.

        chain_2 has 2 vertices (root + leaf) and chain_2 has 1 branch (bullet).
        Vertex 0 (root): append after existing child -> cherry [[], []].
        Vertex 1 (leaf): append after no children -> chain_3 [[[]]].
        """
        result = pgl.product(PT([[]]), PT([[]]))
        expected = (
            ForestSum(((1, PT([[],[]]).as_ordered_forest()),))
            + ForestSum(((1, PT([[[]]]).as_ordered_forest()),))
        )
        self.assertEqual(expected, result)

    def test_product_differs_from_nonplanar(self):
        """Planar GL product splits terms that non-planar GL merges.

        Non-planar: cherry . chain_2 = 1*trident + 2*B+(bullet,chain_2)
          (because B+(chain_2,bullet) == B+(bullet,chain_2) unordered)
        Planar: cherry . chain_2 = 1*trident + 1*B+(bullet,chain_2) + 1*B+(chain_2,bullet)
          (the two orderings are distinct planar trees)
        """
        result = pgl.product(PT([[],[]]), PT([[]]))
        terms = {}
        for c, f in result.term_list:
            terms[f] = terms.get(f, 0) + c
        # All three are distinct planar trees with coefficient 1
        self.assertEqual(terms[PT([[],[],[]]).as_ordered_forest()], 1)
        self.assertEqual(terms[PT([[],[[]]]).as_ordered_forest()], 1)
        self.assertEqual(terms[PT([[[]],[]]).as_ordered_forest()], 1)

    def test_product_planar_sensitivity(self):
        """Grafting preserves planar structure: different orderings give different results."""
        r1 = pgl.product(PT([[],[[]]]), PT([[]]))
        r2 = pgl.product(PT([[[]],[]]), PT([[]]))
        self.assertNotEqual(r1, r2)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            pgl.product('s', PT([]))
        with self.assertRaises(TypeError):
            pgl.product(PT([]), 's')
        with self.assertRaises(TypeError):
            pgl.product(PT(None), PT([]))


class TestAntipode(unittest.TestCase):

    def test_antipode_bullet(self):
        """S(bullet) = bullet."""
        self.assertEqual(_as_fs(PT([])), pgl.antipode(PT([])))

    def test_antipode_chain2(self):
        """S(chain_2) = -chain_2 (primitive => S = -id)."""
        expected = ForestSum(((-1, PT([[]]).as_ordered_forest()),))
        self.assertEqual(expected, pgl.antipode(PT([[]])))

    def test_antipode_chain3(self):
        """S(chain_3) = -chain_3 (primitive)."""
        expected = ForestSum(((-1, PT([[[]]]).as_ordered_forest()),))
        self.assertEqual(expected, pgl.antipode(PT([[[]]])))

    def test_antipode_property(self):
        """Verify sum_Delta S(left) ._PGL right = epsilon(t) . bullet."""
        for t in trees:
            cp = pgl.coproduct(t)
            result = 0
            for c, lf, rf in cp:
                left = lf[0]
                s_left = pgl.antipode(left)
                gl_prod = pgl.product(s_left, rf[0])
                result = result + c * gl_prod
            if pgl.counit(t) == 0:
                self.assertEqual(0, result,
                                 msg=f"Antipode property failed for {t.list_repr}")
            else:
                self.assertEqual(_as_fs(PT([])), result,
                                 msg=f"Antipode property failed for {t.list_repr}")

    def test_antipode_planar_sensitivity(self):
        """Different planar orderings produce different antipodes."""
        s1 = pgl.antipode(PT([[],[[]]]))
        s2 = pgl.antipode(PT([[[]],[]]))
        self.assertNotEqual(s1, s2)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            pgl.antipode('s')


class TestAntipodeInvolution(unittest.TestCase):
    """S^2 = id holds for cocommutative Hopf algebras.
    The planar GL coproduct is cocommutative (S <-> S^c is a bijection
    on subsets), so S^2 = id despite the product being noncommutative."""

    def test_involution_primitives(self):
        """S^2 = id for primitive elements (chains): S = -id so S^2 = id."""
        for t in [PT([]), PT([[]]), PT([[[]]])]:
            s2 = pgl.antipode(pgl.antipode(t))
            self.assertEqual(_as_fs(t), s2,
                             msg=f"S^2({t.list_repr}) should equal t")

    def test_involution_all(self):
        """S^2 = id for all test trees (cocommutative => involution)."""
        for t in trees:
            s2 = pgl.antipode(pgl.antipode(t))
            self.assertEqual(_as_fs(t), s2,
                             msg=f"S^2({t.list_repr}) should equal t")


class TestCounitOfAntipode(unittest.TestCase):
    """Verify epsilon o S = epsilon."""

    def test_counit_of_antipode(self):
        for t in trees:
            s_t = pgl.antipode(t)
            eps_s = 0
            for c, forest in s_t.term_list:
                val = 1
                for tree in forest.tree_list:
                    val *= pgl.counit(tree)
                eps_s += c * val
            self.assertEqual(pgl.counit(t), eps_s,
                             msg=f"eps(S({t.list_repr})) != eps({t.list_repr})")


class TestConvolution(unittest.TestCase):

    def test_counit_conv_identity(self):
        """f * epsilon = f = epsilon * f."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        f_eps = pgl.map_product(f, pgl.counit)
        eps_f = pgl.map_product(pgl.counit, f)
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
        fg_h = pgl.map_product(pgl.map_product(f, g), h)
        f_gh = pgl.map_product(f, pgl.map_product(g, h))
        for t in trees:
            self.assertEqual(fg_h(t), f_gh(t),
                             msg=f"(f*g)*h != f*(g*h) at {t.list_repr}")

    def test_map_product_counit(self):
        """counit * counit = counit."""
        m = pgl.map_product(pgl.counit, pgl.counit)
        for t in trees:
            self.assertEqual(pgl.counit(t), m(t))


class TestMapPower(unittest.TestCase):

    def test_map_power_counit(self):
        """counit^n = counit for all n >= 1."""
        for n in [1, 2, 3]:
            m = pgl.map_power(pgl.counit, n)
            for t in trees:
                self.assertEqual(pgl.counit(t), m(t),
                                 msg=f"counit^{n}({t.list_repr})")

    def test_map_power_zero(self):
        """f^0 = counit."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        m = pgl.map_power(f, 0)
        for t in trees:
            self.assertEqual(pgl.counit(t), m(t),
                             msg=f"f^0({t.list_repr}) should be counit")

    def test_map_power_inverse(self):
        """f^n * f^{-n} = counit."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        f2 = pgl.map_power(f, 2)
        f_neg2 = pgl.map_power(f, -2)
        m = pgl.map_product(f2, f_neg2)
        for t in trees:
            self.assertEqual(pgl.counit(t), m(t),
                             msg=f"f^2 * f^-2 ({t.list_repr})")

    def test_type_error_map_product(self):
        with self.assertRaises(TypeError):
            pgl.map_product('a', pgl.counit)
        with self.assertRaises(TypeError):
            pgl.map_product(pgl.counit, 'b')

    def test_type_error_map_power(self):
        with self.assertRaises(TypeError):
            pgl.map_power('a', 2)
        with self.assertRaises(TypeError):
            pgl.map_power(pgl.counit, 2.5)
