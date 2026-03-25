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
decomposition applied to the planar BCK Hopf algebra.

Unlike the commutative case, the closed-form convolution expressions for
``minus`` and ``plus`` do not transfer to the noncommutative setting.
Instead, ``minus`` and ``plus`` are defined recursively via a grade-based
splitting that directly constructs the factorisation
:math:`\\mathrm{Id} = \\mathrm{Id}^+ \\cdot \\mathrm{Id}^-` in the
planar BCK convolution algebra.
"""

__all__ = ['id_sqrt', 'minus', 'plus']

from .trees import ForestSum, ZERO_FOREST_SUM, _is_scalar
from .pbck.pbck import coproduct_impl
from .generic_algebra import apply_map, forest_apply, func_product
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
The square root of the identity map in the planar BCK Hopf algebra,
:math:`\\mathrm{Id}^{1/2}`. The unique multiplicative map such that
:math:`\\mathrm{Id}^{1/2} \\cdot \\mathrm{Id}^{1/2} = \\mathrm{Id}`
where the product is the convolution in the planar BCK Hopf algebra.
"""


# ---------------------------------------------------------------------------
# Grade-splitting helpers
# ---------------------------------------------------------------------------

def _grade_filter(fs, parity):
    """Extract terms whose forests have node count matching the given parity (0=even, 1=odd)."""
    terms = tuple((c, f) for c, f in fs.term_list if f.nodes() % 2 == parity)
    return ForestSum(terms) if terms else ZERO_FOREST_SUM


# ---------------------------------------------------------------------------
# Recursive plus / minus via grade-splitting
# ---------------------------------------------------------------------------
#
# The factorisation Id = plus * minus (planar BCK convolution) is
# constructed recursively.  For each tree t with Δ(t) = t⊗e + e⊗t + Σ',
#
#   (plus * minus)(t) = plus(t) + minus(t) + Σ'_middle plus(left)·minus(right) = t
#
# so  plus(t) + minus(t) = t - Σ'_middle plus(left)·minus(right)  =: R(t)
#
# We split:  plus(t) = even-grade part of R(t)
#            minus(t) = odd-grade part of R(t)
#
# Base case: plus(e) = minus(e) = e  (empty tree / forest).

@cache
def _remainder(t):
    """Compute R(t) = t - Σ'_{middle Δ} plus(left)·minus(right)."""
    fs_t = t.as_forest_sum()
    cp = coproduct_impl(t)
    all_middle_terms = []
    for c, left, right_forest in cp:
        right = right_forest[0]
        if right.list_repr is None or right.list_repr == t.list_repr:
            continue
        plus_left = forest_apply(left, _planar_plus)
        minus_right = _planar_minus(right)
        product = c * plus_left * minus_right
        if isinstance(product, ForestSum):
            all_middle_terms.extend(product.term_list)
        elif _is_scalar(product):
            pass  # zero contribution
        else:
            all_middle_terms.append((1, product))

    if not all_middle_terms:
        return fs_t

    return ForestSum(fs_t.term_list + tuple((-c, f) for c, f in all_middle_terms)).simplify()


@cache
def _planar_plus(t):
    if t.list_repr is None:
        return t.as_forest_sum()
    return _grade_filter(_remainder(t), 0)


@cache
def _planar_minus(t):
    if t.list_repr is None:
        return t.as_forest_sum()
    return _grade_filter(_remainder(t), 1)


minus = Map(_planar_minus)
minus.__doc__ = """
The minus (odd) part of the identity in the planar BCK Hopf algebra.
Satisfies :math:`\\mathrm{Id}^+ \\cdot \\mathrm{Id}^- = \\mathrm{Id}`
where :math:`\\cdot` is the planar BCK convolution product.
"""

plus = Map(_planar_plus)
plus.__doc__ = """
The plus (even) part of the identity in the planar BCK Hopf algebra.
Satisfies :math:`\\mathrm{Id}^+ \\cdot \\mathrm{Id}^- = \\mathrm{Id}`
where :math:`\\cdot` is the planar BCK convolution product.
"""
