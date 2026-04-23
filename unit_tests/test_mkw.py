"""
Tests for the MKW (Munthe-Kaas-Wright) Hopf algebra (kauri.mkw).

References:
    - H. Munthe-Kaas, W. Wright, "On the Hopf algebraic structure of Lie
      group integrators", Found. Comput. Math. 8 (2008), 227-257.
    - C. Curry, K. Ebrahimi-Fard, D. Manchon, H. Munthe-Kaas,
      "Planarly branched rough paths and rough differential equations
      on homogeneous spaces", J. Differential Equations 269 (2020), 9740-9782.
"""

import unittest
from kauri.trees import (PlanarTree, OrderedForest, ForestSum,
                         TensorProductSum, EMPTY_PLANAR_TREE, EMPTY_ORDERED_FOREST)
from kauri.maps import Map
from kauri.gentrees import planar_trees_of_order
import kauri.mkw as mkw
import kauri.nck as nck
from kauri.mkw.mkw import (shuffle_forests, _shuffle_forestsum_with_forest,
                           _forest_antipode, coproduct_impl, antipode_impl)

PT = PlanarTree
bullet = PT([])
chain2 = PT([[]])
cherry = PT([[], []])
chain3 = PT([[[]]])
# Tree with non-identical children: B+(bullet, chain2)
asym_tree = PT([[], [[]]])


class ShuffleProductTests(unittest.TestCase):

    def test_shuffle_empty_left(self):
        """shuffle(empty, F) = F."""
        f = OrderedForest((bullet, chain2))
        result = shuffle_forests(EMPTY_ORDERED_FOREST, f)
        self.assertEqual(result, f.as_forest_sum())

    def test_shuffle_empty_right(self):
        """shuffle(F, empty) = F."""
        f = OrderedForest((bullet, chain2))
        result = shuffle_forests(f, EMPTY_ORDERED_FOREST)
        self.assertEqual(result, f.as_forest_sum())

    def test_shuffle_single_single(self):
        """shuffle(a, b) = ab + ba (two interleavings)."""
        f1 = bullet.as_ordered_forest()
        f2 = chain2.as_ordered_forest()
        result = shuffle_forests(f1, f2)
        self.assertEqual(len(result.term_list), 2)
        for c, _ in result.term_list:
            self.assertEqual(c, 1)

    def test_shuffle_identical_trees(self):
        """shuffle(a, a) simplifies to 2*aa."""
        f1 = bullet.as_ordered_forest()
        f2 = bullet.as_ordered_forest()
        result = shuffle_forests(f1, f2)
        self.assertEqual(len(result.term_list), 1)
        self.assertEqual(result.term_list[0][0], 2)

    def test_shuffle_commutativity(self):
        """shuffle(F, G) = shuffle(G, F)."""
        f1 = OrderedForest((bullet, chain2))
        f2 = OrderedForest((cherry,))
        r1 = shuffle_forests(f1, f2)
        r2 = shuffle_forests(f2, f1)
        self.assertEqual(r1, r2)

    def test_shuffle_associativity(self):
        """shuffle(shuffle(F, G), H) = shuffle(F, shuffle(G, H))."""
        f1 = bullet.as_ordered_forest()
        f2 = chain2.as_ordered_forest()
        f3 = cherry.as_ordered_forest()

        # (f1 shuffle f2) shuffle f3
        sh12 = shuffle_forests(f1, f2)
        left = _shuffle_forestsum_with_forest(sh12, f3)

        # f1 shuffle (f2 shuffle f3)
        sh23 = shuffle_forests(f2, f3)
        right = ForestSum(())
        for c, f in sh23.term_list:
            term = shuffle_forests(f1, f)
            for sc, sf in term.term_list:
                right = right + ForestSum(((c * sc, sf),))
        right = right.simplify()

        self.assertEqual(left, right)

    def test_shuffle_term_count(self):
        """shuffle of (m) and (n) forests has C(m+n, m) terms before simplification."""
        f1 = OrderedForest((bullet, chain2))   # 2 trees
        f2 = OrderedForest((cherry,))          # 1 tree
        result = shuffle_forests(f1, f2)
        # C(3,2) = 3 interleavings, all distinct since trees differ
        self.assertEqual(len(result.term_list), 3)

    def test_shuffle_public_wrapper(self):
        """shuffle_product accepts PlanarTree arguments."""
        result = mkw.shuffle_product(bullet, chain2)
        self.assertEqual(len(result.term_list), 2)


class CoproductTests(unittest.TestCase):

    def test_coproduct_empty(self):
        """Delta(empty) = empty tensor empty."""
        cp = mkw.coproduct(EMPTY_PLANAR_TREE)
        self.assertEqual(len(cp), 1)

    def test_coproduct_bullet(self):
        """Delta(bullet) = bullet tensor empty + empty tensor bullet."""
        cp = mkw.coproduct(bullet)
        self.assertEqual(len(cp), 2)

    def test_coproduct_chain2(self):
        """Delta(chain2) has 3 terms, same as NCK (single child)."""
        cp = mkw.coproduct(chain2)
        nck_cp = nck.coproduct(chain2)
        self.assertEqual(len(cp), len(nck_cp))

    def test_coproduct_ladder_trees_match_nck(self):
        """For ladder trees (chains), MKW and NCK coproducts are identical."""
        for t in [bullet, chain2, chain3]:
            mkw_cp = mkw.coproduct(t)
            nck_cp = nck.coproduct(t)
            mkw_terms = sorted([(c, l.tree_list, r.tree_list) for c, l, r in mkw_cp],
                               key=str)
            nck_terms = sorted([(c, l.tree_list, r.tree_list) for c, l, r in nck_cp],
                               key=str)
            self.assertEqual(mkw_terms, nck_terms, msg=repr(t.list_repr))

    def test_coproduct_cherry_differs_from_nck(self):
        """Cherry: MKW differs from NCK due to left-admissibility.

        For cherry = B+(•,•), left-admissible cuts allow cutting the left
        root edge {e1} but NOT the right edge {e2} alone.  So MKW has
        1·• ⊗ [[]] while NCK has 2·• ⊗ [[]] (both {e1} and {e2} allowed).
        The (•,•) ⊗ • term has coefficient 1 in both.
        """
        mkw_cp = mkw.coproduct(cherry)
        nck_cp = nck.coproduct(cherry)
        # Both have 4 terms
        self.assertEqual(len(mkw_cp), len(nck_cp))
        # MKW: (bullet,bullet) x bullet has coeff 1 (same-vertex prefix, no shuffle)
        for c, l, r in mkw_cp:
            trees = [t for t in l.tree_list if t.list_repr is not None]
            if len(trees) == 2 and r[0] == bullet:
                self.assertEqual(c, 1)
        # NCK: also coeff 1 on (bullet,bullet) x bullet
        for c, l, r in nck_cp:
            trees = [t for t in l.tree_list if t.list_repr is not None]
            if len(trees) == 2 and r[0] == bullet:
                self.assertEqual(c, 1)
        # MKW has 1·• ⊗ [[]] but NCK has 2·• ⊗ [[]]
        for c, l, r in mkw_cp:
            trees = [t for t in l.tree_list if t.list_repr is not None]
            right_tree = r[0]
            if len(trees) == 1 and trees[0] == bullet and right_tree == PlanarTree([[]]):
                self.assertEqual(c, 1)
        for c, l, r in nck_cp:
            trees = [t for t in l.tree_list if t.list_repr is not None]
            right_tree = r[0]
            if len(trees) == 1 and trees[0] == bullet and right_tree == PlanarTree([[]]):
                self.assertEqual(c, 2)

    def test_coproduct_differs_for_nonidentical_children(self):
        """MKW coproduct differs from NCK for trees with distinct children."""
        # B+(bullet, chain2) — children differ
        mkw_cp = mkw.coproduct(asym_tree)
        nck_cp = nck.coproduct(asym_tree)
        mkw_terms = sorted([(c, l.tree_list, r.tree_list) for c, l, r in mkw_cp],
                           key=str)
        nck_terms = sorted([(c, l.tree_list, r.tree_list) for c, l, r in nck_cp],
                           key=str)
        self.assertNotEqual(mkw_terms, nck_terms)


class CoassociativityTests(unittest.TestCase):
    """Verify coassociativity via convolution associativity on LB characters.

    We check (f*g*h)(t) is the same whether computed as ((f*g)*h)(t) or
    (f*(g*h))(t) using mkw.map_product.  Since MKW convolution is defined
    for **shuffle-symmetric characters** (which is what LB-series
    characters are — every iterated exponential in an RKMK/CF method
    yields a shuffle-symmetric character, and composition preserves
    symmetry), we use LB characters here, not arbitrary scalar maps.
    """

    def _coassoc_check(self, t):
        """Check convolution associativity on a single tree.
        Uses LB characters from predefined CF methods."""
        from kauri import lie_euler, lie_midpoint, cfree_rk3

        f = lie_euler.lb_character()
        g = lie_midpoint.lb_character()
        h = cfree_rk3.lb_character()

        fg = mkw.map_product(f, g)
        gh = mkw.map_product(g, h)

        fg_h = mkw.map_product(fg, h)
        f_gh = mkw.map_product(f, gh)

        self.assertAlmostEqual(fg_h(t), f_gh(t), places=10,
                               msg=f"Coassociativity failed for {t.list_repr}")

    def test_coassociativity_order_1(self):
        for t in planar_trees_of_order(1):
            self._coassoc_check(t)

    def test_coassociativity_order_2(self):
        for t in planar_trees_of_order(2):
            self._coassoc_check(t)

    def test_coassociativity_order_3(self):
        for t in planar_trees_of_order(3):
            self._coassoc_check(t)

    def test_coassociativity_order_4(self):
        for t in planar_trees_of_order(4):
            self._coassoc_check(t)


class AntipodeTests(unittest.TestCase):

    def test_antipode_bullet(self):
        """S(bullet) = -bullet."""
        result = mkw.antipode(bullet)
        expected = ForestSum(((-1, bullet.as_ordered_forest()),))
        self.assertEqual(result, expected)

    def test_antipode_chain2(self):
        """S(chain2) = -chain2 + 2*(bullet,bullet)."""
        result = mkw.antipode(chain2)
        self.assertEqual(len(result.term_list), 2)

    def _verify_antipode_axiom(self, t):
        """Verify mu_shuffle(S tensor id)(Delta(t)) = counit(t).

        Direct check using shuffle multiplication.
        """
        cp = coproduct_impl(t)
        result = ForestSum(())
        for c, left, right in cp:
            s_left = _forest_antipode(left)
            right_tree = right[0]
            if right_tree.list_repr is None:
                result = result + c * s_left
            else:
                term = _shuffle_forestsum_with_forest(s_left, right_tree.as_ordered_forest())
                result = result + c * term
        result = result.simplify()

        if t.list_repr is None:
            expected = EMPTY_ORDERED_FOREST.as_forest_sum()
        else:
            expected = ForestSum(())  # zero
        self.assertEqual(result, expected,
                         msg=f"Antipode axiom failed for {t.list_repr}")

    def test_antipode_axiom_order_1(self):
        for t in planar_trees_of_order(1):
            self._verify_antipode_axiom(t)

    def test_antipode_axiom_order_2(self):
        for t in planar_trees_of_order(2):
            self._verify_antipode_axiom(t)

    def test_antipode_axiom_order_3(self):
        for t in planar_trees_of_order(3):
            self._verify_antipode_axiom(t)

    def test_antipode_axiom_order_4(self):
        for t in planar_trees_of_order(4):
            self._verify_antipode_axiom(t)


class ConvolutionTests(unittest.TestCase):
    """Test convolution product (map_product) for scalar-valued maps."""

    def test_counit_is_identity(self):
        """counit * f = f for scalar-valued f."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        h = mkw.map_product(mkw.counit, f)
        for n in range(1, 4):
            for t in planar_trees_of_order(n):
                self.assertEqual(h(t), f(t), msg=repr(t.list_repr))

    def test_mkw_and_nck_agree_on_ladder_trees(self):
        """MKW and NCK convolutions coincide on ladder (chain) trees —
        their coproducts are identical on ladders since every cut is
        left-admissible (there is only one branch at each vertex)."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        g = Map(lambda t: 1)
        mkw_fg = mkw.map_product(f, g)
        nck_fg = nck.map_product(f, g)
        ladders = [PT([]), PT([[]]), PT([[[]]]), PT([[[[]]]]), PT([[[[[]]]]])]
        for t in ladders:
            self.assertAlmostEqual(mkw_fg(t), nck_fg(t), places=10,
                                   msg=f"MKW/NCK disagree on ladder {t.list_repr}")

    def test_mkw_disagrees_with_nck_on_non_ladders(self):
        """On trees with identical-type siblings, MKW and NCK convolutions
        *should* differ: MKW has coefficient 1 on the bullet⊗chain2 term
        of Delta(cherry) while NCK has coefficient 2."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        g = Map(lambda t: 1)
        mkw_fg = mkw.map_product(f, g)
        nck_fg = nck.map_product(f, g)
        # cherry is the simplest tree where the coproducts differ
        self.assertNotAlmostEqual(mkw_fg(cherry), nck_fg(cherry), places=6,
            msg="MKW and NCK convolutions should differ on cherry")

    def test_map_power_inverse(self):
        """f^(-1) * f = counit for scalar-valued f."""
        f = Map(lambda t: t.nodes() if t.list_repr is not None else 1)
        f_inv = mkw.map_power(f, -1)
        product = mkw.map_product(f_inv, f)
        for n in range(1, 4):
            for t in planar_trees_of_order(n):
                self.assertAlmostEqual(product(t), mkw.counit(t), places=10,
                                       msg=repr(t.list_repr))


class CounitTests(unittest.TestCase):

    def test_counit_empty(self):
        self.assertEqual(mkw.counit(EMPTY_PLANAR_TREE), 1)

    def test_counit_nonempty(self):
        self.assertEqual(mkw.counit(bullet), 0)
        self.assertEqual(mkw.counit(chain2), 0)
        self.assertEqual(mkw.counit(cherry), 0)


# ===========================================================================
# Reference tests: Munthe-Kaas & Wright (2008), arxiv math/0603023
# Table 2: Shuffle products, Table 5: Coproduct, Table 6: Antipode
# ===========================================================================

OF = OrderedForest

# Additional tree definitions for reference tests
chain4   = PT([[[[]]]])
b_cherry = PT([[[], []]])
b_bc2    = PT([[], [[]]])
b_c2b    = PT([[[]], []])
b_bbb    = PT([[], [], []])

# Shorthand list_reprs for expected dicts
B   = bullet.list_repr
C2  = chain2.list_repr
C3  = chain3.list_repr
CH  = cherry.list_repr
C4  = chain4.list_repr
BC  = b_cherry.list_repr
BB  = b_bc2.list_repr
CB  = b_c2b.list_repr
BBB = b_bbb.list_repr


def _fs_dict(fs):
    """Convert ForestSum to {forest_key: coeff} dict for comparison."""
    d = {}
    for c, f in fs.term_list:
        key = tuple(t.list_repr for t in f.tree_list
                    if t.list_repr is not None)
        d[key] = d.get(key, 0) + c
    return {k: v for k, v in d.items() if v != 0}


def _tps_dict(tps):
    """Convert TensorProductSum to {(left_key, right_key): coeff} dict."""
    d = {}
    for c, l, r in tps.term_list:
        lk = tuple(t.list_repr for t in l.tree_list
                   if t.list_repr is not None)
        rk = tuple(t.list_repr for t in r.tree_list
                   if t.list_repr is not None)
        d[(lk, rk)] = d.get((lk, rk), 0) + c
    return {k: v for k, v in d.items() if v != 0}


def _of(*trees_arg):
    """Build an OrderedForest from PlanarTree arguments."""
    if not trees_arg:
        return EMPTY_ORDERED_FOREST
    return OF(trees_arg)


# ---------------------------------------------------------------------------
# Table 2: Shuffle products
# ---------------------------------------------------------------------------

class ShuffleReferenceTests(unittest.TestCase):
    """Shuffle product values from Table 2 of Munthe-Kaas & Wright (2008)."""

    def _check(self, f1, f2, expected):
        result = _fs_dict(shuffle_forests(f1, f2))
        self.assertEqual(result, expected,
                         msg=f"shuffle({f1}, {f2})")

    def test_bullet_bullet(self):
        self._check(_of(bullet), _of(bullet),
                    {(B, B): 2})

    def test_bullet_bullet_bullet(self):
        self._check(_of(bullet), _of(bullet, bullet),
                    {(B, B, B): 3})

    def test_bullet_chain2(self):
        self._check(_of(bullet), _of(chain2),
                    {(C2, B): 1, (B, C2): 1})

    def test_bullet_3bullets(self):
        self._check(_of(bullet), _of(bullet, bullet, bullet),
                    {(B, B, B, B): 4})

    def test_bullet_bullet_chain2(self):
        self._check(_of(bullet), _of(bullet, chain2),
                    {(B, B, C2): 2, (B, C2, B): 1})

    def test_bullet_chain2_bullet(self):
        self._check(_of(bullet), _of(chain2, bullet),
                    {(C2, B, B): 2, (B, C2, B): 1})

    def test_bullet_chain3(self):
        self._check(_of(bullet), _of(chain3),
                    {(B, C3): 1, (C3, B): 1})

    def test_bullet_cherry(self):
        self._check(_of(bullet), _of(cherry),
                    {(B, CH): 1, (CH, B): 1})

    def test_2bullets_2bullets(self):
        self._check(_of(bullet, bullet), _of(bullet, bullet),
                    {(B, B, B, B): 6})

    def test_2bullets_chain2(self):
        self._check(_of(bullet, bullet), _of(chain2),
                    {(B, B, C2): 1, (B, C2, B): 1, (C2, B, B): 1})

    def test_chain2_chain2(self):
        self._check(_of(chain2), _of(chain2),
                    {(C2, C2): 2})


# ---------------------------------------------------------------------------
# Table 5: Coproduct (tree entries)
# ---------------------------------------------------------------------------

class CoproductReferenceTests(unittest.TestCase):
    """MKW coproduct values from Table 5 of Munthe-Kaas & Wright (2008)."""

    def _check(self, tree, expected):
        result = _tps_dict(coproduct_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"coproduct({tree.list_repr})")

    def test_bullet(self):
        self._check(bullet, {
            ((B,), ()): 1,
            ((), (B,)): 1,
        })

    def test_chain2(self):
        self._check(chain2, {
            ((C2,), ()): 1,
            ((B,), (B,)): 1,
            ((), (C2,)): 1,
        })

    def test_chain3(self):
        self._check(chain3, {
            ((C3,), ()): 1,
            ((B,), (C2,)): 1,
            ((C2,), (B,)): 1,
            ((), (C3,)): 1,
        })

    def test_cherry(self):
        self._check(cherry, {
            ((CH,), ()): 1,
            ((B, B), (B,)): 1,
            ((B,), (C2,)): 1,
            ((), (CH,)): 1,
        })

    def test_chain4(self):
        self._check(chain4, {
            ((C4,), ()): 1,
            ((C3,), (B,)): 1,
            ((C2,), (C2,)): 1,
            ((B,), (C3,)): 1,
            ((), (C4,)): 1,
        })

    def test_b_cherry(self):
        self._check(b_cherry, {
            ((BC,), ()): 1,
            ((CH,), (B,)): 1,
            ((B, B), (C2,)): 1,
            ((B,), (C3,)): 1,
            ((), (BC,)): 1,
        })

    def test_b_bc2(self):
        self._check(b_bc2, {
            ((BB,), ()): 1,
            ((B, C2), (B,)): 1,
            ((B, B), (C2,)): 2,
            ((B,), (C3,)): 1,
            ((B,), (CH,)): 1,
            ((), (BB,)): 1,
        })

    def test_b_c2b(self):
        self._check(b_c2b, {
            ((CB,), ()): 1,
            ((C2, B), (B,)): 1,
            ((C2,), (C2,)): 1,
            ((B,), (CH,)): 1,
            ((), (CB,)): 1,
        })

    def test_b_bbb(self):
        self._check(b_bbb, {
            ((BBB,), ()): 1,
            ((B, B, B), (B,)): 1,
            ((B, B), (C2,)): 1,
            ((B,), (CH,)): 1,
            ((), (BBB,)): 1,
        })


# ---------------------------------------------------------------------------
# Table 6: Antipode (tree entries)
# ---------------------------------------------------------------------------

class AntipodeTreeReferenceTests(unittest.TestCase):
    """MKW tree antipode values from Table 6 of Munthe-Kaas & Wright (2008)."""

    def _check(self, tree, expected):
        result = _fs_dict(antipode_impl(tree))
        self.assertEqual(result, expected,
                         msg=f"S({tree.list_repr})")

    def test_bullet(self):
        self._check(bullet, {(B,): -1})

    def test_chain2(self):
        self._check(chain2, {(C2,): -1, (B, B): 2})

    def test_chain3(self):
        self._check(chain3, {
            (C3,): -1,
            (B, C2): 2,
            (C2, B): 2,
            (B, B, B): -6,
        })

    def test_cherry(self):
        self._check(cherry, {
            (CH,): -1,
            (B, C2): 1,
            (C2, B): 1,
            (B, B, B): -3,
        })

    def test_chain4(self):
        self._check(chain4, {
            (C4,): -1,
            (B, C3): 2,
            (C3, B): 2,
            (C2, C2): 2,
            (B, B, C2): -6,
            (B, C2, B): -6,
            (C2, B, B): -6,
            (B, B, B, B): 24,
        })

    def test_b_cherry(self):
        self._check(b_cherry, {
            (BC,): -1,
            (B, C3): 1,
            (C3, B): 1,
            (B, CH): 1,
            (CH, B): 1,
            (B, B, C2): -3,
            (B, C2, B): -3,
            (C2, B, B): -3,
            (B, B, B, B): 12,
        })

    def test_b_bc2(self):
        self._check(b_bc2, {
            (BB,): -1,
            (B, C3): 1,
            (C3, B): 1,
            (B, CH): 1,
            (CH, B): 1,
            (B, B, C2): -2,
            (B, C2, B): -3,
            (C2, B, B): -4,
            (B, B, B, B): 12,
        })

    def test_b_c2b(self):
        self._check(b_c2b, {
            (CB,): -1,
            (B, CH): 1,
            (CH, B): 1,
            (C2, C2): 2,
            (B, B, C2): -4,
            (B, C2, B): -3,
            (C2, B, B): -2,
            (B, B, B, B): 12,
        })

    def test_b_bbb(self):
        self._check(b_bbb, {
            (BBB,): -1,
            (B, CH): 1,
            (CH, B): 1,
            (B, B, C2): -1,
            (B, C2, B): -1,
            (C2, B, B): -1,
            (B, B, B, B): 4,
        })


# ---------------------------------------------------------------------------
# Table 6: Antipode (forest entries)
# ---------------------------------------------------------------------------

class AntipodeForestReferenceTests(unittest.TestCase):
    """MKW forest antipode values from Table 6 of Munthe-Kaas & Wright (2008)."""

    def _check(self, forest, expected):
        result = _fs_dict(_forest_antipode(forest))
        self.assertEqual(result, expected,
                         msg=f"S({[t.list_repr for t in forest.tree_list]})")

    def test_bullet_bullet(self):
        self._check(_of(bullet, bullet), {(B, B): 1})

    def test_bullet_chain2(self):
        self._check(_of(bullet, chain2), {
            (C2, B): 1,
            (B, B, B): -3,
        })

    def test_chain2_bullet(self):
        self._check(_of(chain2, bullet), {
            (B, C2): 1,
            (B, B, B): -3,
        })

    def test_3bullets(self):
        self._check(_of(bullet, bullet, bullet), {(B, B, B): -1})

    def test_bullet_chain3(self):
        self._check(_of(bullet, chain3), {
            (C3, B): 1,
            (B, B, C2): -1,
            (B, C2, B): -2,
            (C2, B, B): -3,
            (B, B, B, B): 12,
        })

    def test_chain3_bullet(self):
        self._check(_of(chain3, bullet), {
            (B, C3): 1,
            (B, B, C2): -3,
            (B, C2, B): -2,
            (C2, B, B): -1,
            (B, B, B, B): 12,
        })

    def test_bullet_cherry(self):
        self._check(_of(bullet, cherry), {
            (CH, B): 1,
            (B, C2, B): -1,
            (C2, B, B): -2,
            (B, B, B, B): 6,
        })

    def test_cherry_bullet(self):
        self._check(_of(cherry, bullet), {
            (B, CH): 1,
            (B, B, C2): -2,
            (B, C2, B): -1,
            (B, B, B, B): 6,
        })

    def test_chain2_chain2(self):
        self._check(_of(chain2, chain2), {
            (C2, C2): 1,
            (B, B, C2): -2,
            (B, C2, B): -2,
            (C2, B, B): -2,
            (B, B, B, B): 12,
        })

    def test_bb_chain2(self):
        self._check(_of(bullet, bullet, chain2), {
            (C2, B, B): -1,
            (B, B, B, B): 4,
        })

    def test_b_chain2_b(self):
        self._check(_of(bullet, chain2, bullet), {
            (B, C2, B): -1,
            (B, B, B, B): 4,
        })

    def test_chain2_bb(self):
        self._check(_of(chain2, bullet, bullet), {
            (B, B, C2): -1,
            (B, B, B, B): 4,
        })

    def test_4bullets(self):
        self._check(_of(bullet, bullet, bullet, bullet),
                    {(B, B, B, B): 1})


if __name__ == '__main__':
    unittest.main()
