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
This module provides instances of ``kauri.Map`` related to the odd-even
decomposition applied to the NCK Hopf algebra
:cite:`aguiar2006combinatorial`.

The ``minus`` map is computed via the convolution formula

.. math::

    \\tau^- = \\mu \\circ (\\overline{S} \\otimes \\mathrm{Id}) \\circ
    \\Delta(\\mathrm{Id}^{1/2}(\\tau))

where :math:`\\overline{S}(\\tau) := (-1)^{|\\tau|}S(\\tau)` and the
coproduct is extended to the :class:`~kauri.trees.ForestSum` returned by
:math:`\\mathrm{Id}^{1/2}`.  The ``plus`` map is then derived
recursively from the factorisation
:math:`\\mathrm{Id} = \\mathrm{Id}^+ \\cdot \\mathrm{Id}^-` in the
NCK convolution algebra:

.. math::

    \\tau^+ = \\tau - \\tau^- - \\sum_{(\\tau)}'
    (\\tau'_{(1)})^+ \\cdot (\\tau'_{(2)})^-
"""

__all__ = ['id_sqrt', 'minus', 'plus']

from .trees import PlanarTree, ForestSum, TensorProductSum, ZERO_FOREST_SUM
from .nck.nck import coproduct_impl, antipode_impl
from .generic_algebra import apply_map, forest_apply, anti_forest_apply, func_product, sign_factor
from .maps import Map
from functools import cache


def _planar_ident(t):
    """Identity returning ForestSum."""
    return t.as_forest_sum()


# ---------------------------------------------------------------------------
# Id^{1/2}
# ---------------------------------------------------------------------------

@cache
def _planar_id_sqrt(t):
    if not isinstance(t, PlanarTree):
        raise TypeError(
            f"planar_oddeven.id_sqrt expects a PlanarTree, not {type(t)}. "
            "Use oddeven.id_sqrt for Tree.")
    fs_t = t.as_forest_sum()
    if t.list_repr is None:
        return fs_t
    if len(t.list_repr) == 1:
        return fs_t * 0.5
    # Convolution square of IDENTITY via planar coproduct
    ident_sq = func_product(t, _planar_ident, _planar_ident, coproduct_impl)
    # Subtract the two edge terms => middle terms only
    out = ident_sq.as_forest_sum() - 2 * fs_t
    # Apply id_sqrt to each tree in the middle terms
    out = apply_map(out, _planar_id_sqrt)
    return (fs_t - out) * 0.5


id_sqrt = Map(_planar_id_sqrt)
id_sqrt.__doc__ = """
The square root of the identity map in the NCK Hopf algebra,
:math:`\\mathrm{Id}^{1/2}`. The unique multiplicative map such that
:math:`\\mathrm{Id}^{1/2} \\cdot \\mathrm{Id}^{1/2} = \\mathrm{Id}`
where the product is the convolution in the NCK Hopf algebra
:cite:`aguiar2006combinatorial`.

**Example usage:**

.. kauri-exec::

    for t in kr.planar_trees_of_order(3):
        kr.display(t, "\\u2192", planar_oddeven.id_sqrt(t), rationalise=True)
"""


# ---------------------------------------------------------------------------
# Extend NCK coproduct to ForestSum
# ---------------------------------------------------------------------------

def _extend_coproduct_to_forestsum(fs):
    """Extend the NCK coproduct from trees to a ForestSum.

    For a forest f = t1·t2·…·tk the coproduct is the ordered product
    Δ(t1)·Δ(t2)·…·Δ(tk), and this is extended linearly over the sum.
    """
    all_terms = []
    for c, forest in fs.term_list:
        cp = None
        for tree in forest.tree_list:
            tree_cp = coproduct_impl(tree)
            cp = tree_cp if cp is None else cp * tree_cp
        if cp is not None:
            all_terms.extend((c * tc, lf, rf) for tc, lf, rf in cp.term_list)
    return TensorProductSum(tuple(all_terms)).simplify()


# ---------------------------------------------------------------------------
# Minus via convolution formula
# ---------------------------------------------------------------------------
#
#   τ⁻ = μ ∘ (S̄ ⊗ Id) ∘ Δ(Id^{1/2}(τ))
#
# where S̄(τ) := (-1)^{|τ|} S(τ) and S is the NCK antipode (an
# anti-homomorphism, so we use anti_forest_apply).

@cache
def _planar_minus(t):
    """Compute Id⁻ via the convolution formula on Δ(Id^{1/2}(t))."""
    if t.list_repr is None:
        return t.as_forest_sum()
    sqrt_t = _planar_id_sqrt(t)
    cp = _extend_coproduct_to_forestsum(sqrt_t)
    terms = []
    for c, left, right in cp.term_list:
        s_left = sign_factor(left) * anti_forest_apply(left, antipode_impl)
        product = c * s_left * right.as_forest_sum()
        terms.extend(product.term_list)
    return ForestSum(tuple(terms)).simplify()


# ---------------------------------------------------------------------------
# Plus via recursive factorisation Id = Id⁺ · Id⁻
# ---------------------------------------------------------------------------
#
# For each tree t with Δ(t) = t⊗e + e⊗t + Σ':
#   plus(t) = t − minus(t) − Σ'_middle plus(left)·minus(right)

@cache
def _planar_plus(t):
    """Derive Id⁺ recursively from the factorisation Id = Id⁺ · Id⁻."""
    if t.list_repr is None:
        return t.as_forest_sum()
    fs_t = t.as_forest_sum()
    cp = coproduct_impl(t)
    middle_terms = []
    for c, left, right_forest in cp.term_list:
        right = right_forest[0]
        if right.list_repr is None or right.list_repr == t.list_repr:
            continue
        product = c * forest_apply(left, _planar_plus) * _planar_minus(right)
        middle_terms.extend(product.term_list)
    middle = ForestSum(tuple(middle_terms))
    return (fs_t - _planar_minus(t) - middle).simplify()


minus = Map(_planar_minus)
minus.__doc__ = """
The minus (odd) part of the identity in the NCK Hopf algebra.
Satisfies :math:`\\mathrm{Id}^+ \\cdot \\mathrm{Id}^- = \\mathrm{Id}`
where :math:`\\cdot` is the NCK convolution product
:cite:`aguiar2006combinatorial`.

**Example usage:**

.. kauri-exec::

    for t in kr.planar_trees_of_order(3):
        kr.display(t, "\\u2192", planar_oddeven.minus(t), rationalise=True)
"""

plus = Map(_planar_plus)
plus.__doc__ = """
The plus (even) part of the identity in the NCK Hopf algebra.
Satisfies :math:`\\mathrm{Id}^+ \\cdot \\mathrm{Id}^- = \\mathrm{Id}`
where :math:`\\cdot` is the NCK convolution product
:cite:`aguiar2006combinatorial`.

**Example usage:**

.. kauri-exec::

    for t in kr.planar_trees_of_order(3):
        kr.display(t, "\\u2192", planar_oddeven.plus(t), rationalise=True)
"""
