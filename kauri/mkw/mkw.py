"""
The MKW (Munthe-Kaas--Wright) Hopf algebra module.
"""
from functools import cache
from itertools import combinations, product as iter_product

from ..maps import Map
from ..trees import (Tree, PlanarTree, OrderedForest,
                     ForestSum, TensorProductSum,
                     EMPTY_ORDERED_FOREST)
from ..generic_algebra import func_product, func_power
from ..nck.nck import coproduct_impl as _nck_coproduct, antipode_impl as _nck_antipode


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


# ---------------------------------------------------------------------------
# Antipode
# ---------------------------------------------------------------------------

def _shuffle_forest_apply(forest, func):
    """Apply func to each tree in forest, combine via iterated shuffle.

    Returns a ForestSum. This is the MKW (shuffle-based) extension of a
    tree-level map to forests, analogous to forest_apply for NCK.
    """
    trees = [t for t in forest.tree_list if t.list_repr is not None]
    if not trees:
        return EMPTY_ORDERED_FOREST.as_forest_sum()

    result = func(trees[0])
    for i in range(1, len(trees)):
        fi = func(trees[i])
        result = _shuffle_forestsums(result, fi)
    return result


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

        # S is a homomorphism (MKW is commutative), extended via shuffle
        s_left = _shuffle_forest_apply(left_forest, antipode_impl)
        # Multiply S(left) by right_tree using shuffle
        right_fs = right_tree.as_ordered_forest()
        term = _shuffle_forestsum_with_forest(s_left, right_fs)
        out = out - c * term

    return out.simplify()


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

counit = Map(counit_impl)
counit.__doc__ = """
The counit :math:`\\varepsilon` of the MKW Hopf algebra.

:type: Map

**Example usage:**

.. kauri-exec::

    print(mkw.counit(PlanarTree(None)))  # Returns 1
    print(mkw.counit(PlanarTree([])))  # Returns 0
"""

def _safe_antipode(t):
    if not isinstance(t, PlanarTree):
        hint = " For non-planar trees, use bck.antipode instead." if isinstance(t, Tree) else ""
        raise TypeError("Argument to mkw.antipode must be a PlanarTree, not " + str(type(t)) + "." + hint)
    return antipode_impl(t)

antipode = Map(_safe_antipode, anti=False)
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

    where :math:`\\mu` is the shuffle product.

    .. note::

        Both maps must be **scalar-valued** (returning numbers, not trees/forests).

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
    # For scalar-valued maps, the MKW and NCK convolutions are identical:
    # the shuffle coefficients in Delta_MKW exactly cancel with the 1/k!
    # factors required by the shuffle-algebra character evaluation.
    # We use the NCK coproduct internally for efficiency and consistency.
    # MKW is commutative, so anti1 is always False (no anti-homomorphism).
    return Map(lambda t: func_product(t, f.func, g.func, _nck_coproduct, anti1=False))


def map_power(f: Map, exponent: int) -> Map:
    """
    Returns the convolution power of a map in the MKW Hopf algebra.

    .. note::

        The map should be **scalar-valued** for iterated powers (exponent > 1 or < 0).

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
    # See map_product for why we use NCK internals for scalar-valued maps.
    return Map(lambda t: func_power(t, f.func, exponent, _nck_coproduct, counit_impl, _nck_antipode, anti1=False))
