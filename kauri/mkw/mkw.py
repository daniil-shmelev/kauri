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
The MKW (Munthe-Kaas--Wright) Hopf algebra module.

The MKW Hopf algebra ``H_MKW = (OF, shuffle, Delta_MKW)`` lives on
ordered (planar) forests with the **commutative shuffle product** and
the left-admissible-cuts coproduct.  Characters of ``H_MKW`` must
therefore be **shuffle-multiplicative**: ``alpha(x shuffle y) = alpha(x)
* alpha(y)``.  The :class:`kauri.maps.Map` objects returned by
:func:`counit`, :func:`antipode`, :func:`map_product` and
:func:`map_power` use ``extension="shuffle"`` so that evaluation on an
ordered forest ``(t_1, ..., t_k)`` picks up the ``1/k!`` shuffle
multinomial denominator.  This is the correct convention for every
LB-series character arising from an RKMK/CF method: a single ``exp(beta
k)`` generates a shuffle-symmetric character, and :func:`map_product`
preserves symmetry, so kauri's MKW characters are always
shuffle-symmetric.  Asymmetric MKW characters are not supported.
"""
from functools import cache
from itertools import combinations, product as iter_product

from ..maps import Map
from ..trees import (Tree, PlanarTree, OrderedForest,
                     ForestSum, TensorProductSum,
                     EMPTY_ORDERED_FOREST, EMPTY_PLANAR_TREE)
from .._protocols import ForestLike
from ..generic_algebra import (
    mkw_apply, mkw_base_char_func,
    mkw_convolution_func, mkw_convolution_power,
)


# ---------------------------------------------------------------------------
# Shuffle product
# ---------------------------------------------------------------------------

@cache
def shuffle_forests(f1: OrderedForest, f2: OrderedForest) -> ForestSum:
    """Shuffle product of two ordered forests.

    Returns the sum of all interleavings of *f1* and *f2* that preserve
    the relative order within each forest.
    """
    trees1 = tuple(t for t in f1.tree_list if t.list_repr is not None)
    trees2 = tuple(t for t in f2.tree_list if t.list_repr is not None)
    m, n = len(trees1), len(trees2)

    if m == 0:
        if n == 0:
            return EMPTY_ORDERED_FOREST.as_forest_sum()
        return OrderedForest(trees2).as_forest_sum()
    if n == 0:
        return OrderedForest(trees1).as_forest_sum()

    terms = []
    for positions in combinations(range(m + n), m):
        merged = [None] * (m + n)
        pos_set = set(positions)
        i1, i2 = 0, 0
        for k in range(m + n):
            if k in pos_set:
                merged[k] = trees1[i1]
                i1 += 1
            else:
                merged[k] = trees2[i2]
                i2 += 1
        terms.append((1, OrderedForest(tuple(merged))))

    return ForestSum(tuple(terms)).simplify()


def _shuffle_forestsum_with_forest(fs: ForestSum, f: OrderedForest) -> ForestSum:
    """Shuffle each forest in a ForestSum with an OrderedForest."""
    terms = []
    for c, forest in fs.term_list:
        sh = shuffle_forests(forest, f)
        for sc, sf in sh.term_list:
            terms.append((c * sc, sf))
    return ForestSum(tuple(terms)).simplify()


def _shuffle_forestsums(fs1: ForestSum, fs2: ForestSum) -> ForestSum:
    """Bilinear extension of shuffle to two ForestSums."""
    terms = []
    for c1, f1 in fs1.term_list:
        for c2, f2 in fs2.term_list:
            sh = shuffle_forests(f1, f2)
            for sc, sf in sh.term_list:
                terms.append((c1 * c2 * sc, sf))
    return ForestSum(tuple(terms)).simplify()


# ---------------------------------------------------------------------------
# Counit
# ---------------------------------------------------------------------------

def counit_impl(t):
    return 1 if t.list_repr is None else 0


# ---------------------------------------------------------------------------
# Coproduct
# ---------------------------------------------------------------------------

@cache
def coproduct_impl(t):
    """MKW coproduct on trees via left-admissible cuts.

    For a tree τ = B₊(τ₁, …, τₖ), the left-admissible cuts require
    that edges cut from the same vertex form a *left prefix* of the
    children at that vertex.  Subtrees pruned from the same vertex
    keep their planar order; forests from different vertices are
    shuffled together.

    For ladder trees (chains where every node has at most one child),
    the MKW and NCK coproducts coincide.
    """
    if not isinstance(t, PlanarTree):
        raise TypeError(
            f"Argument to mkw.coproduct must be a PlanarTree, not {type(t)}. "
            "For non-planar trees, use bck.coproduct instead.")
    if t.list_repr is None:
        return TensorProductSum(((1, EMPTY_ORDERED_FOREST, EMPTY_ORDERED_FOREST),))

    if len(t.list_repr) == 1:
        return TensorProductSum(((1, EMPTY_ORDERED_FOREST, t.as_ordered_forest()),
                (1, t.as_ordered_forest(), EMPTY_ORDERED_FOREST)))

    root_color = t.list_repr[-1]
    children = [PlanarTree(rep) for rep in t.list_repr[:-1]]
    child_coproducts = [coproduct_impl(child) for child in children]
    k = len(children)

    # Separate each child's coproduct into "full prune" (τᵢ ⊗ 1)
    # and "internal" terms (everything else).
    # Full-prune = right side is the empty tree, i.e. the root→child
    # edge is cut.  Internal = right side is non-empty.
    child_internal = []
    for cp in child_coproducts:
        internal = [term for term in cp.term_list
                    if term[2][0].list_repr is not None]
        child_internal.append(internal)

    # Term: t ⊗ 1 (always present)
    raw_terms = [(1, OrderedForest((t,)), EMPTY_ORDERED_FOREST)]

    # Iterate over m = 0..k: fully prune the left prefix of m children.
    # Left-admissibility requires that root-level cuts form a left
    # prefix, so children[0..m-1] are fully pruned and children[m..k-1]
    # may only have internal cuts (no root-edge cuts).
    for m in range(k + 1):
        # Prefix forest: fully-pruned children in planar order
        if m > 0:
            prefix_forest = OrderedForest(tuple(children[:m]))
        else:
            prefix_forest = EMPTY_ORDERED_FOREST

        remaining = [child_internal[i] for i in range(m, k)]

        if not remaining:
            # All children fully pruned → trunk is just the root
            right = PlanarTree((root_color,))
            raw_terms.append((1, prefix_forest, right.as_ordered_forest()))
            continue

        # Iterate over combinations of internal terms from remaining children
        for picks in iter_product(*remaining):
            lefts = []
            right_repr_children = []
            coeff = 1
            for c, left_forest, right_forest in picks:
                right_tree = right_forest[0]
                coeff *= c
                lefts.append(left_forest)
                if right_tree.list_repr is not None:
                    right_repr_children.append(right_tree.list_repr)
            right_repr_children.append(root_color)
            right = PlanarTree(tuple(right_repr_children))

            # Shuffle internal left forests from different children
            shuffled = lefts[0].as_forest_sum()
            for i in range(1, len(lefts)):
                shuffled = _shuffle_forestsum_with_forest(shuffled, lefts[i])

            # Shuffle the root-level prefix with the internal forests
            # (different vertices → shuffle)
            if m > 0:
                shuffled = _shuffle_forestsum_with_forest(shuffled, prefix_forest)

            for sc, sf in shuffled.term_list:
                raw_terms.append((coeff * sc, sf, right.as_ordered_forest()))

    return TensorProductSum(tuple(raw_terms)).simplify()


def _b_plus(trees: tuple) -> PlanarTree:
    """Graft the given trees onto a new root of colour 0.  This is the
    B+ operation of Munthe-Kaas--Wright (and Connes-Kreimer); ``trees``
    must be a tuple of non-empty PlanarTrees."""
    return PlanarTree(tuple(t.list_repr for t in trees) + (0,))


def _basis_aware_func(m: "Map"):
    """Return ``m.func`` if ``m`` is already a basis-aware MKW Map (i.e. its
    ``func`` handles both trees and ordered forests); otherwise wrap
    ``m.func`` as an MKW base character via :func:`mkw_base_char_func`.
    Used by :func:`map_product` and :func:`map_power` to accept either
    base characters or previously-composed characters interchangeably."""
    if getattr(m, "_mkw_basis_aware", False):
        return m.func
    return mkw_base_char_func(m.func)


def _as_basis_aware_map(func, *, cache_key: bool = True) -> "Map":
    """Build a ``Map(extension='shuffle')`` from a basis-aware ``func`` and
    tag it ``_mkw_basis_aware = True`` so that subsequent convolutions
    skip redundant wrapping."""
    m = Map(func, extension="shuffle")
    m._mkw_basis_aware = True
    return m


# ---------------------------------------------------------------------------
# Forest coproduct — the canonical extension of Delta_tree to ordered forests
# via the B+/B- recursion from Munthe-Kaas & Wright (2008, eq. derived from
# Definition 5 / eq. 1031–1035 of the paper).  For an ordered forest
# omega = (t_1, ..., t_k),
#
#     Delta_forest(omega) = (id tensor B-) ( Delta_tree(B+(omega)) - B+(omega) tensor 1 )
#
# where B+(omega) is the tree obtained by grafting all trees of omega onto a
# common new root (colour 0), and B-(tau) extracts the children of tau as an
# ordered forest.  This is the coassociative extension used by the paper's
# Table 5 and the only one that gives associative convolution.  Reduces to
# coproduct_impl(tau) when omega is a single-tree forest.
# ---------------------------------------------------------------------------

@cache
def forest_coproduct_impl(forest: OrderedForest) -> TensorProductSum:
    """MKW coproduct on an ordered forest, via the B+/B- recursion.

    For empty or single-tree forests, reduces to the tree coproduct (with
    appropriate wrapping).  For multi-tree forests, constructs the parent
    tree ``B+(omega)``, takes the tree coproduct, removes the ``B+(omega)
    otimes 1`` term, and applies ``id otimes B-`` — yielding
    ``Delta_forest(omega)`` with ordered forests on both sides.
    """
    trees = tuple(t for t in forest.tree_list if t.list_repr is not None)

    if not trees:
        # Delta(unit) = unit tensor unit
        return TensorProductSum(
            ((1, EMPTY_ORDERED_FOREST, EMPTY_ORDERED_FOREST),))

    if len(trees) == 1:
        # Single-tree forest: promote the tree coproduct (right side is a
        # 1-tree forest already, matching the forest-coproduct convention).
        return coproduct_impl(trees[0])

    parent = _b_plus(trees)
    parent_cp = coproduct_impl(parent)

    raw_terms = []
    for c, left, right in parent_cp.term_list:
        right_tree = right[0]

        # The B+(omega) tensor 1 term has left = (parent,) and right is empty.
        # Skip it per the formula.
        if right_tree.list_repr is None:
            left_trees = tuple(
                t for t in left.tree_list if t.list_repr is not None)
            if len(left_trees) == 1 and left_trees[0] == parent:
                continue

        # Apply (id tensor B-): replace right_tree by its children as a forest.
        if right_tree.list_repr is None:
            new_right = EMPTY_ORDERED_FOREST
        else:
            child_reprs = right_tree.list_repr[:-1]
            if child_reprs:
                new_right = OrderedForest(
                    tuple(PlanarTree(r) for r in child_reprs))
            else:
                # B-(leaf) = empty forest
                new_right = EMPTY_ORDERED_FOREST

        raw_terms.append((c, left, new_right))

    return TensorProductSum(tuple(raw_terms)).simplify()


# ---------------------------------------------------------------------------
# Antipode
# ---------------------------------------------------------------------------

@cache
def _forest_antipode(forest):
    """MKW antipode applied to an ordered forest basis element.

    For single-tree forests, delegates to ``antipode_impl``.
    For multi-tree forests, uses the forest coproduct

    .. math::

        \\Delta_{\\text{forest}}(\\omega)
          = (\\mathrm{id} \\otimes B_-)
            \\bigl(\\Delta(B_+(\\omega)) - B_+(\\omega) \\otimes 1\\bigr)

    and the recursive formula

    .. math::

        S(\\omega) = -\\omega
          - \\sum' S(\\text{left}) \\shuffle B_-(\\text{right}).
    """
    trees = tuple(t for t in forest.tree_list if t.list_repr is not None)

    if not trees:
        return EMPTY_ORDERED_FOREST.as_forest_sum()

    if len(trees) == 1:
        return antipode_impl(trees[0])

    parent = _b_plus(trees)
    cp = coproduct_impl(parent)

    the_forest = OrderedForest(trees)
    out = ForestSum(((-1, the_forest),))

    for c, left, right in cp.term_list:
        right_tree = right[0]

        # Skip τ ⊗ 1  and  1 ⊗ τ
        if right_tree.list_repr is None or right_tree == parent:
            continue

        # B₋(right_tree): extract children as an ordered forest
        right_children = tuple(
            PlanarTree(r) for r in right_tree.list_repr[:-1])

        if not right_children:
            # B₋(leaf) = empty → this is the ω ⊗ 1 term, skip
            continue

        right_forest = OrderedForest(right_children)

        # Recursively compute S(left_forest)
        s_left = _forest_antipode(left)

        # S(left) ⊔⊔ B₋(right)
        term = _shuffle_forestsum_with_forest(s_left, right_forest)
        out = out - c * term

    return out.simplify()


@cache
def antipode_impl(t):
    if t.list_repr is None:
        return ForestSum(((1, EMPTY_ORDERED_FOREST),))

    if len(t.list_repr) == 1:
        return ForestSum(((-1, t.as_ordered_forest()),))

    cp = coproduct_impl(t)
    out = ForestSum(((-1, t.as_ordered_forest()),))

    for c, left_forest, right_forest in cp:
        right_tree = right_forest[0]
        if right_tree.list_repr is None or right_tree == t:
            continue

        s_left = _forest_antipode(left_forest)
        right_fs = right_tree.as_ordered_forest()
        term = _shuffle_forestsum_with_forest(s_left, right_fs)
        out = out - c * term

    return out.simplify()


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def _counit_basis(x):
    """Counit on any MKW basis element: 1 on the unit, 0 elsewhere."""
    if isinstance(x, ForestLike):
        non_empty = [t for t in x.tree_list if t.list_repr is not None]
        return 1 if not non_empty else 0
    return counit_impl(x)

counit = _as_basis_aware_map(_counit_basis)
counit.__doc__ = """
The counit :math:`\\varepsilon` of the MKW Hopf algebra.

:type: Map

**Example usage:**

.. kauri-exec::

    print(mkw.counit(PlanarTree(None)))  # Returns 1
    print(mkw.counit(PlanarTree([])))  # Returns 0
"""

def _safe_antipode(x):
    """Basis-aware MKW antipode: dispatches to :func:`antipode_impl` on
    trees and to :func:`_forest_antipode` on ordered forests."""
    if isinstance(x, PlanarTree):
        return antipode_impl(x)
    if isinstance(x, OrderedForest):
        return _forest_antipode(x)
    hint = " For non-planar trees, use bck.antipode instead." if isinstance(x, Tree) else ""
    raise TypeError(
        "Argument to mkw.antipode must be a PlanarTree or OrderedForest, "
        "not " + str(type(x)) + "." + hint)

antipode = _as_basis_aware_map(_safe_antipode)
antipode.__doc__ = """
The antipode :math:`S` of the MKW Hopf algebra.

Since the MKW algebra is commutative (the shuffle product is commutative),
the antipode is a **homomorphism**: :math:`S(f_1 \\shuffle f_2) = S(f_1) \\shuffle S(f_2)`.

:type: Map

**Example usage:**

.. kauri-exec::

    t = PlanarTree([[[]],[]])
    kr.display(mkw.antipode(t))
"""


def coproduct(t: PlanarTree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta` of the MKW Hopf algebra.

    The MKW coproduct uses the shuffle product to combine left factors
    from different children's coproducts, unlike the NCK coproduct which
    uses concatenation.  For ladder trees (chains), the two coincide.

    :param t: planar tree
    :type t: PlanarTree
    :rtype: TensorProductSum

    **Example usage:**

    .. kauri-exec::

        t = PlanarTree([[],[]])
        kr.display(mkw.coproduct(t))
    """
    if not isinstance(t, PlanarTree):
        hint = " For non-planar trees, use bck.coproduct instead." if isinstance(t, Tree) else ""
        raise TypeError("Argument to mkw.coproduct must be a PlanarTree, not " + str(type(t)) + "." + hint)
    return coproduct_impl(t)


def shuffle_product(f1, f2) -> ForestSum:
    """
    The shuffle product of two ordered forests (or planar trees).

    Returns a :class:`~kauri.trees.ForestSum` containing all interleavings
    of the two input forests that preserve the relative order within each.

    :param f1: first forest (or planar tree)
    :param f2: second forest (or planar tree)
    :rtype: ForestSum

    **Example usage:**

    .. kauri-exec::

        t1 = PlanarTree([])
        t2 = PlanarTree([[]])
        kr.display(mkw.shuffle_product(t1, t2))
    """
    if isinstance(f1, PlanarTree):
        f1 = f1.as_ordered_forest()
    if isinstance(f2, PlanarTree):
        f2 = f2.as_ordered_forest()
    return shuffle_forests(f1, f2)


def map_product(f: Map, g: Map) -> Map:
    """
    Returns the convolution product of scalar-valued maps in the MKW
    Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{MKW}(t)

    where :math:`\\mu` is the shuffle product and
    :math:`\\Delta_{MKW}` is the left-admissible-cuts coproduct.  The
    inputs are interpreted as **shuffle-multiplicative** scalar characters,
    and the returned :class:`Map` carries ``extension="shuffle"`` so that
    subsequent evaluations on ordered forests apply the shuffle-symmetric
    ``1/k!`` extension.

    .. note::

        Both maps must be **scalar-valued** (returning numbers, not
        trees/forests) and interpreted as shuffle-symmetric characters.
        The true-solution LB character on a Lie group has different tree
        values than the flat-space B-series character; e.g.
        ``alpha_exact(cherry) = 1/6`` under MKW, not ``1/3``.  See
        ``unit_tests/test_cf_methods.py::TestLBCompositionSemantics``
        for the order-4 table.

    :param f: f
    :type f: Map
    :param g: g
    :type g: Map
    :rtype: Map

    **Example usage:**

    .. kauri-exec::

        f = Map(lambda x: 1 if x.list_repr is None else 0)
        g = mkw.map_product(f, f)
        print(g(PlanarTree([[]])))
    """
    if not (isinstance(f, Map) and isinstance(g, Map)):
        raise TypeError("Arguments in mkw.map_product must be of type Map, not "
                        + str(type(f)) + " and " + str(type(g)))

    f_fn = _basis_aware_func(f)
    g_fn = _basis_aware_func(g)

    def conv(x):
        return mkw_convolution_func(
            x, f_fn, g_fn, coproduct_impl, forest_coproduct_impl)
    return _as_basis_aware_map(conv)


def map_power(f: Map, exponent: int) -> Map:
    """
    Returns the convolution power of a scalar-valued map in the MKW Hopf
    algebra.  The result carries ``extension="shuffle"`` so that forest
    evaluations obey the shuffle-symmetric :math:`1/k!` extension.

    .. note::

        The map should be **scalar-valued** and interpreted as a
        shuffle-symmetric character.  See :func:`map_product` for the
        conventions used.

    :param f: f
    :type f: Map
    :param exponent: exponent
    :type exponent: int
    :rtype: Map

    **Example usage:**

    .. kauri-exec::

        f = Map(lambda x: 1 if x.list_repr is None else x.nodes())
        f_sq = mkw.map_power(f, 2)
        print(f_sq(PlanarTree([[]])))
    """
    if not isinstance(f, Map):
        raise TypeError("f must be a Map, not " + str(type(f)))
    if not isinstance(exponent, int):
        raise TypeError("exponent must be an int, not " + str(type(exponent)))

    f_fn = _basis_aware_func(f)

    def pow_fn(x):
        return mkw_convolution_power(
            x, f_fn, exponent,
            coproduct_impl, forest_coproduct_impl,
            _counit_basis, _safe_antipode)
    return _as_basis_aware_map(pow_fn)
