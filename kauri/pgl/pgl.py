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
The planar Grossman-Larson Hopf algebra module
"""
from functools import cache
from itertools import product as iter_product, combinations
from collections import defaultdict

from ..maps import Map
from ..trees import (Tree, PlanarTree, NoncommutativeForest, OrderedForest, ForestSum,
                     EMPTY_PLANAR_TREE, EMPTY_ORDERED_FOREST)
from ..generic_algebra import forest_apply, forest_sum_apply
from .._protocols import ForestLike, ForestSumLike


# ---------------------------------------------------------------------------
# Internal helpers: grafting product
# ---------------------------------------------------------------------------

def _vertex_child_counts(repr_tuple):
    """Return list of child counts indexed by pre-order vertex index."""
    counts = []
    def traverse(r):
        children = r[:-1]
        counts.append(len(children))
        for child in children:
            traverse(child)
    traverse(repr_tuple)
    return counts


def _planar_graft_helper(s_repr, vertex_new_branches, current_idx):
    """Recursively graft branches at specified vertices with specified positions.

    vertex_new_branches[v] = (new_reprs, positions) where positions is a sorted
    tuple of indices in range(d+m) specifying where new branches go in the
    merged child list.
    """
    root_color = s_repr[-1]
    original_children = s_repr[:-1]

    # Recursively process original children (advancing vertex counter)
    next_idx = current_idx + 1
    processed = []
    for child_repr in original_children:
        new_child, next_idx = _planar_graft_helper(child_repr, vertex_new_branches, next_idx)
        processed.append(new_child)

    # Merge with new branches if any
    info = vertex_new_branches.get(current_idx)
    if info is None:
        return tuple(processed) + (root_color,), next_idx

    new_reprs, positions = info
    d = len(processed)
    m = len(new_reprs)
    pos_set = set(positions)

    merged = []
    orig_idx = 0
    new_idx = 0
    for i in range(d + m):
        if i in pos_set:
            merged.append(new_reprs[new_idx])
            new_idx += 1
        else:
            merged.append(processed[orig_idx])
            orig_idx += 1

    return tuple(merged) + (root_color,), next_idx


def _pgl_product_trees(s, t):
    """Compute planar GL grafting product s . t for two PlanarTrees.

    Returns a list of PlanarTrees (with repetitions). For each assignment
    of t's branches to vertices of s, and each interleaving at each vertex,
    one result tree is produced.
    """
    branch_reprs = t.list_repr[:-1]
    k = len(branch_reprs)
    n = s.nodes()
    child_counts = _vertex_child_counts(s.list_repr)

    results = []
    for assignment in iter_product(range(n), repeat=k):
        # Group branches by vertex, preserving order
        vb = defaultdict(list)
        for i, v in enumerate(assignment):
            vb[v].append(branch_reprs[i])

        # For each vertex receiving branches, enumerate interleavings
        vertices = sorted(vb.keys())
        if not vertices:
            # k=0: t is bullet, product is just s
            results.append(PlanarTree(s.list_repr))
            continue

        interleaving_options = []
        for v in vertices:
            d = child_counts[v]
            m = len(vb[v])
            options = [(v, tuple(vb[v]), pos) for pos in combinations(range(d + m), m)]
            interleaving_options.append(options)

        for combo in iter_product(*interleaving_options):
            vnb = {}
            for v, new_reprs, positions in combo:
                vnb[v] = (new_reprs, positions)
            result_repr, _ = _planar_graft_helper(s.list_repr, vnb, 0)
            results.append(PlanarTree(result_repr))

    return results


def _pgl_product_linear(s_forestsum, t):
    """Extend the planar GL product linearly: ForestSum ._PGL PlanarTree -> ForestSum."""
    terms = []
    for c, f in s_forestsum.term_list:
        s = f.tree_list[0]
        if s.list_repr is None:
            continue
        result_trees = _pgl_product_trees(s, t)
        for tree in result_trees:
            terms.append((c, tree.as_ordered_forest()))
    if not terms:
        return ForestSum(((0, EMPTY_ORDERED_FOREST),)).simplify()
    return ForestSum(tuple(terms)).simplify()


# ---------------------------------------------------------------------------
# Counit
# ---------------------------------------------------------------------------

def counit_impl(t):
    if t.list_repr is None:
        return 0
    return 1 if len(t.list_repr) == 1 else 0


# ---------------------------------------------------------------------------
# Coproduct
# ---------------------------------------------------------------------------

def _simplify_coproduct(terms):
    """Merge equal (OrderedForest, PlanarTree) terms by summing coefficients."""
    merged = {}
    for c, left, right in terms:
        key = (left, right)
        merged[key] = merged.get(key, 0) + c
    return tuple((c, l, r) for (l, r), c in merged.items() if c != 0)


@cache
def coproduct_impl(t):
    if t.list_repr is None:
        raise TypeError("PGL coproduct is not defined for the empty tree")
    root_color = t.list_repr[-1]
    children = t.list_repr[:-1]
    k = len(children)

    raw_terms = []
    for mask in iter_product([0, 1], repeat=k):
        left_children = tuple(children[i] for i in range(k) if mask[i] == 1)
        right_children = tuple(children[i] for i in range(k) if mask[i] == 0)
        left_tree = PlanarTree(left_children + (root_color,))
        right_tree = PlanarTree(right_children + (root_color,))
        raw_terms.append((1, OrderedForest((left_tree,)), right_tree))

    return _simplify_coproduct(raw_terms)


# ---------------------------------------------------------------------------
# Antipode
# ---------------------------------------------------------------------------

@cache
def antipode_impl(t):
    # Handle empty tree (not part of PGL, but needed for Map infrastructure)
    if t.list_repr is None:
        return ForestSum(((1, PlanarTree([]).as_ordered_forest()),))

    # S(bullet) = bullet
    if len(t.list_repr) == 1:
        return ForestSum(((1, t.as_ordered_forest()),))

    # Recursive: S(t) = -t - sum_{proper} S(left) ._PGL right
    cp = coproduct_impl(t)
    out = ForestSum(((-1, t.as_ordered_forest()),))

    for c, left_forest, right_tree in cp:
        left = left_forest.tree_list[0]

        # Skip trivial terms: t tensor bullet and bullet tensor t
        if left == t or right_tree == t:
            continue

        s_left = antipode_impl(left)  # ForestSum
        pgl_prod = _pgl_product_linear(s_left, right_tree)
        out = out - c * pgl_prod

    return out.simplify()


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------

def _planar_func_product(t, func1, func2, coproduct):
    """Convolution product of scalar-valued maps using the PGL coproduct."""
    cp = coproduct(t)
    if len(cp) == 0:
        return 0
    c0, left0, right0 = cp[0]
    out = c0 * forest_apply(left0, func1) * func2(right0)
    for c, left, right in cp[1:]:
        out += c * forest_apply(left, func1) * func2(right)
    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()
    return out


def _pgl_conv_inverse(func):
    """Return the PGL convolution inverse of func (recursive formula)."""
    memo = {}
    bullet = PlanarTree([])

    def inv(t):
        if t.list_repr is None:
            return func(t)
        key = t.list_repr
        if key in memo:
            return memo[key]

        if len(t.list_repr) == 1:
            result = 1 / func(t)
        else:
            cp = coproduct_impl(t)
            inv_bullet = inv(bullet)
            result = 0
            for c, lf, right_tree in cp:
                if right_tree == t:
                    continue
                left = lf.tree_list[0]
                result += c * func(left) * inv(right_tree)
            result = -inv_bullet * result

        memo[key] = result
        return result

    return inv


def _planar_func_power(t, func, exponent, coproduct, counit, antipode):
    """Convolution power of a scalar-valued map using the PGL coproduct."""
    if exponent == 0:
        return counit(t)
    elif exponent == 1:
        return func(t)
    elif exponent < 0:
        def m(x):
            return _planar_func_power(x, func, -exponent, coproduct, counit, antipode)
        res = forest_sum_apply(antipode(t), m)
    else:
        def m(x):
            return _planar_func_power(x, func, exponent - 1, coproduct, counit, antipode)
        res = _planar_func_product(t, func, m, coproduct)

    if isinstance(res, (ForestLike, ForestSumLike)):
        res = res.simplify()
    return res


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

counit = Map(counit_impl)
counit.__doc__ = """
The counit :math:`\\varepsilon_{PGL}` of the planar Grossman-Larson Hopf algebra.

:type: Map

Example usage::

    from kauri.trees import PlanarTree
    import kauri.pgl as pgl

    pgl.counit(PlanarTree([]))    # Returns 1
    pgl.counit(PlanarTree([[]]))  # Returns 0
"""

antipode = Map(antipode_impl)
antipode.__doc__ = """
The antipode :math:`S_{PGL}` of the planar Grossman-Larson Hopf algebra.

:type: Map

Example usage::

    from kauri.trees import PlanarTree
    import kauri.pgl as pgl

    pgl.antipode(PlanarTree([[]]))  # Returns -1 * [/]
"""


def coproduct(t: PlanarTree) -> tuple:
    """
    The coproduct :math:`\\Delta_{PGL}` of the planar Grossman-Larson Hopf algebra.

    For a tree :math:`t = B_+(t_1, \\ldots, t_k)`:

    .. math::

        \\Delta_{PGL}(t) = \\sum_{S \\subseteq \\{1,\\ldots,k\\}}
            B_+(t_i : i \\in S) \\otimes B_+(t_j : j \\notin S)

    where sibling order is preserved on both sides.

    :param t: planar tree
    :type t: PlanarTree
    :rtype: tuple[tuple[int, OrderedForest, PlanarTree], ...]

    Example usage::

        from kauri.trees import PlanarTree
        import kauri.pgl as pgl

        pgl.coproduct(PlanarTree([]))   # bullet tensor bullet
        pgl.coproduct(PlanarTree([[]]))  # bullet tensor / + / tensor bullet
    """
    if not isinstance(t, PlanarTree):
        hint = " For non-planar trees, use gl.coproduct instead." if isinstance(t, Tree) else ""
        raise TypeError("Argument to pgl.coproduct must be a PlanarTree, not " + str(type(t)) + "." + hint)
    if t.list_repr is None:
        raise TypeError("PGL coproduct is not defined for the empty tree")
    return coproduct_impl(t)


def product(s, t):
    """
    The planar Grossman-Larson grafting product.

    For trees :math:`s` and :math:`t = B_+(b_1, \\ldots, b_k)`, sums over all
    ways of attaching each :math:`b_i` to a vertex of :math:`s` at all possible
    insertion positions among existing children.

    Also accepts a ForestSum as the first argument (linear extension).

    :param s: left operand
    :type s: PlanarTree or ForestSum
    :param t: right operand
    :type t: PlanarTree
    :rtype: ForestSum

    Example usage::

        from kauri.trees import PlanarTree
        import kauri.pgl as pgl

        pgl.product(PlanarTree([[]]), PlanarTree([[]]))
    """
    if isinstance(s, ForestSum) and isinstance(t, PlanarTree):
        if t.list_repr is None:
            raise TypeError("PGL product is not defined for the empty tree")
        return _pgl_product_linear(s, t)
    if isinstance(s, PlanarTree) and isinstance(t, PlanarTree):
        if s.list_repr is None or t.list_repr is None:
            raise TypeError("PGL product is not defined for the empty tree")
        result_trees = _pgl_product_trees(s, t)
        terms = tuple((1, tree.as_ordered_forest()) for tree in result_trees)
        return ForestSum(terms).simplify()
    hint = " For non-planar trees, use gl.product instead." if isinstance(s, Tree) or isinstance(t, Tree) else ""
    raise TypeError(
        "PGL product expects (PlanarTree, PlanarTree) or (ForestSum, PlanarTree), got ("
        + str(type(s)) + ", " + str(type(t)) + ")." + hint
    )


def map_product(f: Map, g: Map) -> Map:
    """
    Returns the convolution product of scalar-valued maps in the planar GL
    Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{PGL}(t)

    :param f: f
    :type f: Map
    :param g: g
    :type g: Map
    :rtype: Map
    """
    if not (isinstance(f, Map) and isinstance(g, Map)):
        raise TypeError("Arguments in pgl.map_product must be of type Map, not "
                        + str(type(f)) + " and " + str(type(g)))
    return Map(lambda t: _planar_func_product(t, f.func, g.func, coproduct_impl))


def map_power(f: Map, exponent: int) -> Map:
    """
    Returns the convolution power of a scalar-valued map in the planar GL
    Hopf algebra.

    :param f: f
    :type f: Map
    :param exponent: exponent
    :type exponent: int
    :rtype: Map
    """
    if not isinstance(f, Map):
        raise TypeError("f must be a Map, not " + str(type(f)))
    if not isinstance(exponent, int):
        raise TypeError("exponent must be an int, not " + str(type(exponent)))
    if exponent >= 0:
        return Map(lambda t: _planar_func_power(t, f.func, exponent, coproduct_impl, counit_impl, antipode_impl))
    f_inv = _pgl_conv_inverse(f.func)
    return Map(lambda t: _planar_func_power(t, f_inv, -exponent, coproduct_impl, counit_impl, antipode_impl))
