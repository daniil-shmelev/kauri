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
The planar BCK Hopf algebra module
"""
from functools import cache
from itertools import product as iter_product

from ..maps import Map
from ..trees import (PlanarTree, NoncommutativeForest, OrderedForest, ForestSum,
                     EMPTY_PLANAR_TREE, EMPTY_ORDERED_FOREST)
from ..generic_algebra import forest_apply, forest_sum_apply
from .._protocols import ForestLike, ForestSumLike


# ---------------------------------------------------------------------------
# Counit
# ---------------------------------------------------------------------------

def counit_impl(t):
    return 1 if t.list_repr is None else 0


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
        return ((1, EMPTY_ORDERED_FOREST, EMPTY_PLANAR_TREE),)

    if len(t.list_repr) == 1:
        return ((1, EMPTY_ORDERED_FOREST, t),
                (1, t.as_ordered_forest(), EMPTY_PLANAR_TREE))

    root_color = t.list_repr[-1]
    children = [PlanarTree(rep) for rep in t.list_repr[:-1]]
    child_coproducts = [coproduct_impl(child) for child in children]

    raw_terms = [(1, OrderedForest((t,)), EMPTY_PLANAR_TREE)]

    for picks in iter_product(*child_coproducts):
        left_trees = []
        right_repr_children = []
        coeff = 1
        for c, left_forest, right_tree in picks:
            coeff *= c
            left_trees.extend(left_forest.tree_list)
            if right_tree.list_repr is not None:
                right_repr_children.append(right_tree.list_repr)
        right_repr_children.append(root_color)

        left = OrderedForest(tuple(left_trees)).simplify()
        right = PlanarTree(tuple(right_repr_children))
        raw_terms.append((coeff, left, right))

    return _simplify_coproduct(raw_terms)


# ---------------------------------------------------------------------------
# Antipode
# ---------------------------------------------------------------------------

def _forest_sum_mul_tree(fs, tree):
    """Multiply a ForestSum on the right by a PlanarTree (ordered concatenation)."""
    new_terms = []
    for c, forest in fs.term_list:
        new_forest = NoncommutativeForest(forest.tree_list + (tree,)).simplify()
        new_terms.append((c, new_forest))
    return ForestSum(tuple(new_terms)).simplify()


def _anti_forest_apply(f, func):
    """Apply func to each tree in forest f in reversed order (anti-homomorphism).

    In a noncommutative Hopf algebra the antipode is an anti-algebra-homomorphism:
    S(t1 * t2 * ... * tk) = S(tk) * ... * S(t2) * S(t1).
    """
    trees = list(reversed(f.tree_list))
    it = iter(trees)
    out = func(next(it))
    for t in it:
        out = out * func(t)
    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()
    return out


@cache
def antipode_impl(t):
    if t.list_repr is None:
        return ForestSum(((1, EMPTY_ORDERED_FOREST),))

    if len(t.list_repr) == 1:
        return ForestSum(((-1, t.as_ordered_forest()),))

    cp = coproduct_impl(t)
    out = ForestSum(((-1, t.as_ordered_forest()),))

    for c, left_forest, right_tree in cp:
        if right_tree.list_repr is None or right_tree == t:
            continue

        s_left = _anti_forest_apply(left_forest, antipode_impl)
        term = _forest_sum_mul_tree(s_left, right_tree)
        out = out - c * term

    return out.simplify()


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------

def _planar_func_product(t, func1, func2, coproduct):
    """Convolution product of scalar-valued maps using the planar coproduct."""
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


def _planar_func_power(t, func, exponent, coproduct, counit, antipode):
    """Convolution power of a scalar-valued map using the planar coproduct."""
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
The counit :math:`\\varepsilon` of the planar BCK Hopf algebra.

:type: Map

Example usage::

    from kauri.trees import PlanarTree
    import kauri.pbck as pbck

    pbck.counit(PlanarTree(None))  # Returns 1
    pbck.counit(PlanarTree([]))    # Returns 0
"""

antipode = Map(antipode_impl)
antipode.__doc__ = """
The antipode :math:`S` of the planar BCK Hopf algebra.

:type: Map

Example usage::

    from kauri.trees import PlanarTree
    import kauri.pbck as pbck

    pbck.antipode(PlanarTree([]))    # Returns -1 * [bullet]
    pbck.antipode(PlanarTree([[]]))  # Returns -1 * [/] + 1 * [bullet, bullet]
"""


def coproduct(t: PlanarTree) -> tuple:
    """
    The coproduct :math:`\\Delta` of the planar BCK Hopf algebra.

    Returns a tuple of ``(coefficient, OrderedForest, PlanarTree)`` triples
    representing the coproduct as a sum of tensor products.

    :param t: planar tree
    :type t: PlanarTree
    :rtype: tuple[tuple[int, OrderedForest, PlanarTree], ...]

    Example usage::

        from kauri.trees import PlanarTree
        import kauri.pbck as pbck

        pbck.coproduct(PlanarTree(None))  # Returns ((1, empty_forest, empty_tree),)
        pbck.coproduct(PlanarTree([]))    # Returns unit coproduct terms
    """
    if not isinstance(t, PlanarTree):
        raise TypeError("Argument to pbck.coproduct must be a PlanarTree, not " + str(type(t)))
    return coproduct_impl(t)


def map_product(f: Map, g: Map) -> Map:
    """
    Returns the convolution product of scalar-valued maps in the planar BCK
    Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta(t)

    :param f: f
    :type f: Map
    :param g: g
    :type g: Map
    :rtype: Map

    Example usage::

        from kauri.trees import PlanarTree
        from kauri.maps import Map
        import kauri.pbck as pbck

        f = Map(lambda x: 1 if x.list_repr is None else 0)
        g = pbck.map_product(f, f)
    """
    if not (isinstance(f, Map) and isinstance(g, Map)):
        raise TypeError("Arguments in pbck.map_product must be of type Map, not "
                        + str(type(f)) + " and " + str(type(g)))
    return Map(lambda t: _planar_func_product(t, f.func, g.func, coproduct_impl))


def map_power(f: Map, exponent: int) -> Map:
    """
    Returns the convolution power of a scalar-valued map in the planar BCK
    Hopf algebra.

    :param f: f
    :type f: Map
    :param exponent: exponent
    :type exponent: int
    :rtype: Map

    Example usage::

        from kauri.trees import PlanarTree
        from kauri.maps import Map
        import kauri.pbck as pbck

        f = Map(lambda x: 1 if x.list_repr is None else x.nodes())
        f_sq = pbck.map_power(f, 2)
    """
    if not isinstance(f, Map):
        raise TypeError("f must be a Map, not " + str(type(f)))
    if not isinstance(exponent, int):
        raise TypeError("exponent must be an int, not " + str(type(exponent)))
    return Map(lambda t: _planar_func_power(t, f.func, exponent, coproduct_impl, counit_impl, antipode_impl))
