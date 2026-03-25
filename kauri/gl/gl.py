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
The Grossman-Larson Hopf algebra module
"""
from functools import cache
from itertools import product as iter_product
from collections import defaultdict

from ..maps import Map
from ..trees import (Tree, Forest, ForestSum, TensorProductSum,
                     PlanarTree, EMPTY_FOREST, ZERO_FOREST_SUM, _is_scalar)
from ..generic_algebra import func_product, func_power


def _gl_combine(a, b):
    """GL product that falls back to scalar multiplication for scalar arguments."""
    if _is_scalar(a) or _is_scalar(b):
        return a * b
    return product(a, b)


# ---------------------------------------------------------------------------
# Internal helpers: grafting product
# ---------------------------------------------------------------------------

def _graft_helper(s_repr, vertex_branches, current_idx):
    """Recursively graft branches at specified vertices of s.

    Vertices are numbered in pre-order (root = current_idx).

    Parameters
    ----------
    s_repr : tuple
        Labelled tuple representation of tree s.
    vertex_branches : dict
        Maps vertex index -> list of branch list_reprs to graft there.
    current_idx : int
        Pre-order index of the current node.

    Returns
    -------
    (new_repr, next_idx) : (tuple, int)
    """
    root_color = s_repr[-1]
    children = s_repr[:-1]

    new_branches = vertex_branches.get(current_idx, [])

    next_idx = current_idx + 1
    processed = []
    for child_repr in children:
        new_child, next_idx = _graft_helper(child_repr, vertex_branches, next_idx)
        processed.append(new_child)

    all_children = tuple(processed) + tuple(new_branches)
    return all_children + (root_color,), next_idx


def _gl_product_trees(s, t):
    """Compute GL grafting product s · t for two Trees.

    Returns a list of Trees (with repetitions corresponding to
    different vertex assignments).  The list has |V(s)|^k entries,
    where k is the number of children of t's root.
    """
    branch_reprs = t.list_repr[:-1]  # child representations of t
    k = len(branch_reprs)
    n = s.nodes()

    results = []
    for assignment in iter_product(range(n), repeat=k):
        vb = defaultdict(list)
        for i, v in enumerate(assignment):
            vb[v].append(branch_reprs[i])
        result_repr, _ = _graft_helper(s.list_repr, vb, 0)
        results.append(Tree(result_repr))

    return results


def _gl_product_linear(s_forestsum, t):
    """Extend the GL product linearly: ForestSum ·_GL Tree -> ForestSum."""
    terms = []
    for c, f in s_forestsum.term_list:
        s = f.tree_list[0]
        if s.list_repr is None:
            continue
        result_trees = _gl_product_trees(s, t)
        for tree in result_trees:
            terms.append((c, tree))
    if not terms:
        return ZERO_FOREST_SUM
    return ForestSum(tuple(terms)).simplify()


# ---------------------------------------------------------------------------
# Counit
# ---------------------------------------------------------------------------

def counit_impl(t):
    # Return 1 if t is the single-vertex tree, otherwise 0
    if t.list_repr is None:
        return 0
    return 1 if len(t.list_repr) == 1 else 0


# ---------------------------------------------------------------------------
# Coproduct
# ---------------------------------------------------------------------------

@cache
def coproduct_impl(t):
    if not isinstance(t, Tree):
        raise TypeError(
            f"Argument to gl.coproduct must be a Tree, not {type(t).__name__}. "
            "For planar trees, use pgl.coproduct instead.")
    if t.list_repr is None:
        raise TypeError("GL coproduct is not defined for the empty tree")
    # GL coproduct: enumerate all 2^k subsets of children of the root.
    # For t with children c_1, ..., c_k and root color r:
    #   Delta(t) = sum_{S subset {1..k}} B+(c_i : i in S) tensor B+(c_j : j not in S)
    root_color = t.list_repr[-1]
    children = t.list_repr[:-1]
    k = len(children)

    terms = []
    for mask in iter_product([0, 1], repeat=k):
        left_children = tuple(children[i] for i in range(k) if mask[i] == 1)
        right_children = tuple(children[i] for i in range(k) if mask[i] == 0)
        left_tree = Tree(left_children + (root_color,))
        right_tree = Tree(right_children + (root_color,))
        terms.append((1, left_tree.as_forest(), right_tree.as_forest()))

    return TensorProductSum(tuple(terms)).simplify()


# ---------------------------------------------------------------------------
# Antipode
# ---------------------------------------------------------------------------

@cache
def antipode_impl(t):
    # Handle empty tree (not part of GL, but needed for Map infrastructure)
    if t.list_repr is None:
        return Tree([]).as_forest_sum()

    # S(bullet) = bullet
    if len(t.list_repr) == 1:
        return t.as_forest_sum()

    # Recursive: S(t) = -t - sum_{proper} S(left) .GL right
    cp = coproduct_impl(t)
    out = -t.as_forest_sum()
    for c, left_forest, right_forest in cp:
        left = left_forest.tree_list[0]
        right = right_forest.tree_list[0]

        # Skip the two trivial terms: t tensor bullet and bullet tensor t
        if left.equals(t) or right.equals(t):
            continue

        s_left = antipode_impl(left)  # ForestSum of tree terms
        gl_prod = _gl_product_linear(s_left, right)
        out = out - c * gl_prod

    return out.simplify()


# ---------------------------------------------------------------------------
# Convolution inverse (GL-specific)
# ---------------------------------------------------------------------------

def _gl_conv_inverse(func):
    """Return a function that evaluates the GL convolution inverse of func.

    Only works for scalar-valued maps (characters).
    """
    memo = {}
    bullet = Tree([])

    def inv(t):
        if t.list_repr is None:
            return func(t)
        key = t.sorted_list_repr()
        if key in memo:
            return memo[key]

        if len(t.list_repr) == 1:
            result = 1 / func(t)
        else:
            cp = coproduct_impl(t)
            inv_bullet = inv(bullet)
            result = 0
            for c, lf, rf in cp:
                right = rf.tree_list[0]
                if right.equals(t):
                    continue
                left = lf.tree_list[0]
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
The counit :math:`\\varepsilon_{GL}` of the Grossman-Larson Hopf algebra.

:type: Map

Example usage::

    import kauri as kr
    import kauri.gl as gl

    gl.counit(kr.Tree([])) # Returns 1
    gl.counit(kr.Tree([[]])) # Returns 0
"""

def _safe_antipode(t):
    if not isinstance(t, Tree):
        hint = " For planar trees, use pgl.antipode instead." if isinstance(t, PlanarTree) else ""
        raise TypeError("Argument to gl.antipode must be a Tree, not " + str(type(t)) + "." + hint)
    return antipode_impl(t)

antipode = Map(_safe_antipode)
antipode.__doc__ = """
The antipode :math:`S_{GL}` of the Grossman-Larson Hopf algebra.

:type: Map

Example usage::

    import kauri as kr
    import kauri.gl as gl

    t = kr.Tree([[],[]])
    gl.antipode(t)
"""


def coproduct(t: Tree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta_{GL}` of the Grossman-Larson Hopf algebra.

    For a tree :math:`t = B_+(t_1, \\ldots, t_k)`:

    .. math::

        \\Delta_{GL}(t) = \\sum_{S \\subseteq \\{1,\\ldots,k\\}} B_+(t_i : i \\in S) \\otimes B_+(t_j : j \\notin S)

    :param t: tree
    :type t: Tree
    :rtype: TensorProductSum

    Example usage::

        import kauri as kr
        import kauri.gl as gl

        gl.coproduct(kr.Tree([])) # Returns 1 [] tensor []
        gl.coproduct(kr.Tree([[]])) # Returns 1 [] tensor [[]]+1 [[]] tensor []
    """
    if not isinstance(t, Tree):
        hint = " For planar trees, use pgl.coproduct instead." if isinstance(t, PlanarTree) else ""
        raise TypeError("Argument to gl.coproduct must be a Tree, not " + str(type(t)) + "." + hint)
    if t.list_repr is None:
        raise TypeError("GL coproduct is not defined for the empty tree")
    return coproduct_impl(t)


def product(s, t):
    """
    The Grossman-Larson grafting product.

    For trees :math:`s` and :math:`t = B_+(b_1, \\ldots, b_k)`, sums over all
    ways of attaching each :math:`b_i` to a vertex of :math:`s`:

    .. math::

        s \\cdot_{GL} t = \\sum_{f: \\{1,\\ldots,k\\} \\to V(s)} \\mathrm{graft}(s, b_1, \\ldots, b_k, f)

    Extends bilinearly to ForestSum arguments.

    :param s: left operand
    :type s: Tree or ForestSum
    :param t: right operand
    :type t: Tree or ForestSum
    :rtype: ForestSum

    Example usage::

        import kauri as kr
        import kauri.gl as gl

        gl.product(kr.Tree([[]]), kr.Tree([[]])) # Returns 1 [[], []] + 1 [[[]]]
    """
    if isinstance(s, Tree):
        if s.list_repr is None:
            raise TypeError("GL product is not defined for the empty tree")
        s = s.as_forest_sum()
    if isinstance(t, Tree):
        if t.list_repr is None:
            raise TypeError("GL product is not defined for the empty tree")
        t = t.as_forest_sum()
    if isinstance(s, ForestSum) and isinstance(t, ForestSum):
        result = ZERO_FOREST_SUM
        for c, f in t.term_list:
            tree = f.tree_list[0]
            if tree.list_repr is None:
                continue
            result = result + c * _gl_product_linear(s, tree)
        return result.simplify()
    hint = " For planar trees, use pgl.product instead." if isinstance(s, PlanarTree) or isinstance(t, PlanarTree) else ""
    raise TypeError(
        "GL product expects Tree or ForestSum arguments, got ("
        + str(type(s)) + ", " + str(type(t)) + ")." + hint
    )


def map_product(f: Map, g: Map) -> Map:
    """
    Returns the convolution product of scalar-valued maps in the GL Hopf algebra,
    defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{GL} (t)

    :param f: f
    :type f: Map
    :param g: g
    :type g: Map
    :rtype: Map

    Example usage::

        import kauri as kr
        import kauri.gl as gl

        f = kr.Map(lambda x : 1 if len(x.list_repr) == 1 else 0)
        g = gl.map_product(f, f)
    """
    if not (isinstance(f, Map) and isinstance(g, Map)):
        raise TypeError("Arguments in gl.map_product must be of type Map, not "
                        + str(type(f)) + " and " + str(type(g)))
    return Map(lambda t: func_product(t, f.func, g.func, coproduct_impl, singleton_reduce=True, product=_gl_combine))


def map_power(f: Map, exponent: int) -> Map:
    """
    Returns the convolution power of a map in the GL Hopf algebra.

    For negative exponents, the convolution inverse is computed via the
    recursive formula specific to the GL coproduct, then raised to the
    corresponding positive power. Negative exponents require scalar-valued maps.

    :param f: f
    :type f: Map
    :param exponent: exponent
    :type exponent: int
    :rtype: Map

    Example usage::

        import kauri as kr
        import kauri.gl as gl

        f = kr.Map(lambda x : 1 if len(x.list_repr) == 1 else 0)
        f_sq = gl.map_power(f, 2)
    """
    if not isinstance(f, Map):
        raise TypeError("f must be a Map, not " + str(type(f)))
    if not isinstance(exponent, int):
        raise TypeError("exponent must be an int, not " + str(type(exponent)))
    if exponent >= 0:
        return Map(lambda t: func_power(t, f.func, exponent, coproduct_impl, counit_impl, antipode_impl, singleton_reduce=True, product=_gl_combine))
    test_val = f.func(Tree([]))
    if not _is_scalar(test_val):
        raise TypeError(
            "gl.map_power with negative exponent requires a scalar-valued map. "
            "Got " + str(type(test_val)) + " for the single-vertex tree. "
            "For tree-valued maps, use bck.map_power instead."
        )
    f_inv = _gl_conv_inverse(f.func)
    return Map(lambda t: func_power(t, f_inv, -exponent, coproduct_impl, counit_impl, antipode_impl, singleton_reduce=True, product=_gl_combine))
