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
The BCK Hopf algebra module
"""
from functools import cache
from ..maps import Map
from ..trees import (Tree, PlanarTree, TensorProductSum,
                     EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM)
from ..generic_algebra import forest_apply


def counit_impl(t):
    # Return 1 if t is the empty tree, otherwise 0
    return 1 if t.list_repr is None else 0

@cache
def antipode_impl(t):
    if t.list_repr is None:
        return EMPTY_FOREST_SUM # Antipode of empty tree is the empty tree
    cp = coproduct_impl(t)
    out = -t.as_forest_sum() # First term, -t
    for c, branches, subtree_ in cp: # Remaining terms
        subtree = subtree_[0] # Convert from Forest to Tree
        if subtree.equals(t) or subtree.equals(EMPTY_TREE):
            continue # We've already included the -t term at the start, so move on
        out = out - c * forest_apply(branches, antipode_impl) * subtree

    return out.simplify()

@cache
def coproduct_impl(t):
    if not isinstance(t, Tree):
        hint = " Use nck.coproduct for planar trees, or nck.map_power/nck.map_product for Map operations." if isinstance(t, PlanarTree) else ""
        raise TypeError("BCK coproduct expects a Tree, not " + str(type(t)) + "." + hint)
    # This follows the recursive definition of https://arxiv.org/pdf/hep-th/9808042
    # using B_- and B_+
    if t == Tree(None):
        return TensorProductSum(( (1, EMPTY_FOREST, EMPTY_FOREST), )) # Tree(None) @ Tree(None)
    if len(t.list_repr) == 1:
        return TensorProductSum(( (1, EMPTY_FOREST, t.as_forest()), (1, t.as_forest(), EMPTY_FOREST) )) # Tree(None) @ t + t @ Tree(None)

    root_color = t.list_repr[-1]
    branches = t.unjoin()

    cp_prod = 1
    for subtree in branches:
        cp = coproduct_impl(subtree)
        cp_prod = cp_prod * cp

    # Return t \otimes \emptyset + (id \otimes B_+)[\Delta(B_-(t))]
    out = t @ Tree(None) + TensorProductSum(tuple((c, f1, f2.join(root_color)) for c, f1, f2 in cp_prod))
    return out.simplify()

counit = Map(counit_impl)
counit.__doc__ = """
The counit :math:`\\varepsilon_{BCK}` of the BCK Hopf algebra.

:type: Map

**Example usage:**

.. kauri-exec::

    print(bck.counit(Tree(None)))  # Returns 1
    print(bck.counit(Tree([])))  # Returns 0
"""

def _safe_antipode(t):
    if not isinstance(t, Tree):
        hint = " For planar trees, use nck.antipode instead." if isinstance(t, PlanarTree) else ""
        raise TypeError("Argument to bck.antipode must be a Tree, not " + str(type(t)) + "." + hint)
    return antipode_impl(t)

antipode = Map(_safe_antipode)
antipode.__doc__ = """
The antipode :math:`S_{BCK}` of the BCK Hopf algebra.

:type: Map

**Example usage:**

.. kauri-exec::

    t = Tree([[[]],[]])
    kr.display(bck.antipode(t))
"""

def coproduct(t : Tree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta_{BCK}` of the BCK Hopf algebra.

    :param t: tree
    :type t: Tree
    :rtype: TensorProductSum

    **Example usage:**

    .. kauri-exec::

        t = Tree([[[]],[]])
        kr.display(bck.coproduct(t))
    """
    if not isinstance(t, Tree):
        hint = " For planar trees, use nck.coproduct instead." if isinstance(t, PlanarTree) else ""
        raise TypeError("Argument to bck.coproduct must be a Tree, not " + str(type(t)) + "." + hint)
    return coproduct_impl(t)

def map_product(f : Map, g : Map) -> Map:
    """
    Returns the product of maps in the BCK Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

    .. note::
        `bck.map_product(f,g)` is equivalent to the Map operator `f * g`

    :param f: f
    :type f: Map
    :param g: g
    :type g: Map
    :rtype: Map

    **Example usage:**

    .. kauri-exec::

        f = bck.map_product(ident, bck.antipode)
        print(f(Tree([[]])))
    """
    if not (isinstance(f, Map) and isinstance(g, Map)):
        raise TypeError("Arguments in bck.map_product must be of type Map, not " + str(type(f)) + " and " + str(type(g)))
    return f * g

def map_power(f : Map, exponent : int) -> Map:
    """
    Returns the power of a map in the BCK Hopf algebra, where the product of functions is defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

    and negative powers are defined as :math:`f^{-n} = (f \\circ S_{BCK})^n`,
    where :math:`S_{BCK}` is the BCK antipode.

    .. note::
        `bck.map_power(f, n)` is equivalent to the Map operator `f ** n`

    :param f: f
    :type f: Map
    :param exponent: exponent
    :type exponent: int

    **Example usage:**

    .. kauri-exec::

        S = bck.map_power(ident, -1)  # antipode
        print(S(Tree([[]])))
    """
    if not isinstance(f, Map):
        raise TypeError("f must be a Map, not " + str(type(f)))
    if not isinstance(exponent, int):
        raise TypeError("exponent must be an int, not " + str(type(exponent)))
    return f ** exponent
