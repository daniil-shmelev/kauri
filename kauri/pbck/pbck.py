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
from ..trees import (Tree, PlanarTree, NoncommutativeForest, OrderedForest, ForestSum,
                     TensorProductSum, EMPTY_ORDERED_FOREST)
from ..generic_algebra import forest_apply, func_product, func_power, anti_forest_apply


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
    if t.list_repr is None:
        return TensorProductSum(((1, EMPTY_ORDERED_FOREST, EMPTY_ORDERED_FOREST),))

    if len(t.list_repr) == 1:
        return TensorProductSum(((1, EMPTY_ORDERED_FOREST, t.as_ordered_forest()),
                (1, t.as_ordered_forest(), EMPTY_ORDERED_FOREST)))

    root_color = t.list_repr[-1]
    children = [PlanarTree(rep) for rep in t.list_repr[:-1]]
    child_coproducts = [coproduct_impl(child) for child in children]

    raw_terms = [(1, OrderedForest((t,)), EMPTY_ORDERED_FOREST)]

    for picks in iter_product(*child_coproducts):
        left_trees = []
        right_repr_children = []
        coeff = 1
        for c, left_forest, right_forest in picks:
            right_tree = right_forest[0]
            coeff *= c
            left_trees.extend(left_forest.tree_list)
            if right_tree.list_repr is not None:
                right_repr_children.append(right_tree.list_repr)
        right_repr_children.append(root_color)

        left = OrderedForest(tuple(left_trees)).simplify()
        right = PlanarTree(tuple(right_repr_children))
        raw_terms.append((coeff, left, right.as_ordered_forest()))

    return TensorProductSum(tuple(raw_terms)).simplify()


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

        s_left = anti_forest_apply(left_forest, antipode_impl)
        term = _forest_sum_mul_tree(s_left, right_tree)
        out = out - c * term

    return out.simplify()


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

antipode = Map(antipode_impl, anti=True)
antipode.__doc__ = """
The antipode :math:`S` of the planar BCK Hopf algebra.

Since the planar BCK algebra is noncommutative, the antipode is an
anti-homomorphism: :math:`S(t_1 t_2) = S(t_2) S(t_1)`. This map uses
``anti=True`` to ensure forests are processed in reversed order.

:type: Map

Example usage::

    from kauri.trees import PlanarTree
    import kauri.pbck as pbck

    pbck.antipode(PlanarTree([]))    # Returns -1 * [bullet]
    pbck.antipode(PlanarTree([[]]))  # Returns -1 * [/] + 1 * [bullet, bullet]
"""


def coproduct(t: PlanarTree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta` of the planar BCK Hopf algebra.

    :param t: planar tree
    :type t: PlanarTree
    :rtype: TensorProductSum

    Example usage::

        from kauri.trees import PlanarTree
        import kauri.pbck as pbck

        pbck.coproduct(PlanarTree(None))  # Returns 1 [] tensor []
        pbck.coproduct(PlanarTree([]))    # Returns unit coproduct terms
    """
    if not isinstance(t, PlanarTree):
        hint = " For non-planar trees, use bck.coproduct instead." if isinstance(t, Tree) else ""
        raise TypeError("Argument to pbck.coproduct must be a PlanarTree, not " + str(type(t)) + "." + hint)
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
    return Map(lambda t: func_product(t, f.func, g.func, coproduct_impl))


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
    return Map(lambda t: func_power(t, f.func, exponent, coproduct_impl, counit_impl, antipode_impl))
