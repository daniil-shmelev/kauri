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
from itertools import product as iter_product
from collections import defaultdict

from ..maps import Map
from ..trees import (Tree, PlanarTree, NoncommutativeForest, OrderedForest, ForestSum,
                     TensorProductSum, EMPTY_ORDERED_FOREST)
from ..generic_algebra import forest_apply, func_product, func_power
from ..gl.gl import _graft_helper


# ---------------------------------------------------------------------------
# Internal helpers: grafting product
# ---------------------------------------------------------------------------

def _pgl_product_trees(s, t):
    """Compute planar GL grafting product s . t for two PlanarTrees.

    Returns a list of PlanarTrees (with repetitions).  For each assignment
    of t's branches to vertices of s, one result tree is produced by
    appending the assigned branches to the right of existing children.
    The list has |V(s)|^k entries where k = number of children of t's root.
    """
    branch_reprs = t.list_repr[:-1]
    k = len(branch_reprs)
    n = s.nodes()

    results = []
    for assignment in iter_product(range(n), repeat=k):
        vb = defaultdict(list)
        for i, v in enumerate(assignment):
            vb[v].append(branch_reprs[i])
        result_repr, _ = _graft_helper(s.list_repr, vb, 0)
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

@cache
def coproduct_impl(t):
    if t.list_repr is None:
        raise TypeError("PGL coproduct is not defined for the empty tree")
    root_color = t.list_repr[-1]
    children = t.list_repr[:-1]
    k = len(children)

    terms = []
    for mask in iter_product([0, 1], repeat=k):
        left_children = tuple(children[i] for i in range(k) if mask[i] == 1)
        right_children = tuple(children[i] for i in range(k) if mask[i] == 0)
        left_tree = PlanarTree(left_children + (root_color,))
        right_tree = PlanarTree(right_children + (root_color,))
        terms.append((1, left_tree.as_ordered_forest(), right_tree.as_ordered_forest()))

    return TensorProductSum(tuple(terms)).simplify()


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

    for c, left_forest, right_forest in cp:
        left = left_forest[0]
        right = right_forest[0]

        # Skip trivial terms: t tensor bullet and bullet tensor t
        if left == t or right == t:
            continue

        s_left = antipode_impl(left)  # ForestSum
        pgl_prod = _pgl_product_linear(s_left, right)
        out = out - c * pgl_prod

    return out.simplify()


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------

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
            for c, lf, rf in cp:
                right = rf[0]
                if right == t:
                    continue
                left = lf[0]
                result += c * func(left) * inv(right)
            result = -inv_bullet * result

        memo[key] = result
        return result

    return inv


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

antipode = Map(antipode_impl, anti=True)
antipode.__doc__ = """
The antipode :math:`S_{PGL}` of the planar Grossman-Larson Hopf algebra.

Since the planar GL algebra is noncommutative, the antipode is an
anti-homomorphism: :math:`S(t_1 t_2) = S(t_2) S(t_1)`. This map uses
``anti=True`` to ensure forests are processed in reversed order.

:type: Map

Example usage::

    from kauri.trees import PlanarTree
    import kauri.pgl as pgl

    pgl.antipode(PlanarTree([[]]))  # Returns -1 * [/]
"""


def coproduct(t: PlanarTree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta_{PGL}` of the planar Grossman-Larson Hopf algebra.

    For a tree :math:`t = B_+(t_1, \\ldots, t_k)`:

    .. math::

        \\Delta_{PGL}(t) = \\sum_{S \\subseteq \\{1,\\ldots,k\\}}
            B_+(t_i : i \\in S) \\otimes B_+(t_j : j \\notin S)

    where sibling order is preserved on both sides.

    :param t: planar tree
    :type t: PlanarTree
    :rtype: TensorProductSum

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
    ways of assigning each :math:`b_i` to a vertex of :math:`s`, appending
    assigned branches to the right of existing children.

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
    return Map(lambda t: func_product(t, f.func, g.func, coproduct_impl))


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
        return Map(lambda t: func_power(t, f.func, exponent, coproduct_impl, counit_impl, antipode_impl))
    f_inv = _pgl_conv_inverse(f.func)
    return Map(lambda t: func_power(t, f_inv, -exponent, coproduct_impl, counit_impl, antipode_impl))
