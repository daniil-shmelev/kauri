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
Utility functions for dealing with generic Hopf algebras on trees.

Two forest-extension conventions coexist here:

* **Concatenation** (``forest_apply``, ``func_product``, ``func_power``) —
  ``alpha(t_1 * ... * t_k) = alpha(t_1) * ... * alpha(t_k)``.  Correct for
  non-commutative algebras (BCK, NCK, CEM, GL, PGL).
* **MKW** (``mkw_apply``, ``mkw_convolution_func``, ``mkw_convolution_power``,
  ``mkw_base_char_func``, ``mkw_shuffle_symmetric_value``) — base characters
  extend to ordered forests via the shuffle-symmetric formula
  ``alpha(t_1, ..., t_k) = prod alpha(t_i) / k!``, while convolution results
  evaluate on forests through the paper's forest coproduct (implemented in
  ``kauri.mkw.forest_coproduct_impl``).  This split is essential for
  associative iterated convolution on the MKW Hopf algebra.
"""
import math

from ._protocols import ForestLike, ForestSumLike


def sign_factor(t) -> int:
    """Return the scalar sign ``(-1)^|t|``."""
    return 1 if t.nodes() % 2 == 0 else -1

def forest_apply(f, func):
    # Apply a function func multiplicatively to a forest f
    it = iter(f.tree_list)
    out = func(next(it))
    for t in it:
        out = out * func(t)

    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()
    return out

def anti_forest_apply(f, func):
    # Apply func to each tree in forest f in reversed order (anti-homomorphism).
    # S(t1 * t2 * ... * tk) = S(tk) * ... * S(t2) * S(t1)
    it = reversed(f.tree_list)
    out = func(next(it))
    for t in it:
        out = out * func(t)
    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()
    return out

def forest_sum_apply(fs, func):
    # Applies a function func linearly and multiplicatively to a forest sum fs
    out = 0
    for c, f in fs.term_list:
        it = iter(f.tree_list)
        term = func(next(it))
        for t in it:
            term = term * func(t)
        out += c * term

    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()
    return out

def anti_forest_sum_apply(fs, func):
    # Same as forest_sum_apply but with reversed tree order (anti-homomorphism)
    out = 0
    for c, f in fs.term_list:
        it = reversed(f.tree_list)
        term = func(next(it))
        for t in it:
            term = term * func(t)
        out += c * term

    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()
    return out

def apply_map(t, func, anti=False):
    # Applies a function func as a linear multiplicative map to a Forest or ForestSum t
    if isinstance(t, ForestLike):
        return anti_forest_apply(t, func) if anti else forest_apply(t, func)
    if isinstance(t, ForestSumLike):
        return anti_forest_sum_apply(t, func) if anti else forest_sum_apply(t, func)
    return func(t)


def mkw_apply(x, char_fn):
    """Evaluate an MKW-character function ``char_fn`` on an arbitrary MKW
    basis element ``x`` (Tree / PlanarTree / OrderedForest / ForestSum).

    The key contract: ``char_fn`` MUST itself handle both trees and
    ordered forests — either because it is a base character produced by
    :func:`mkw_base_char_func` (shuffle-symmetric Pi/k! extension to forests)
    or because it is a convolution result whose forest values come from
    the paper's Delta_forest recursion.  In either case ``char_fn`` is a
    single callable from basis-element to scalar; this helper merely
    extends it linearly over a ForestSum.
    """
    if isinstance(x, ForestSumLike):
        out = 0
        for c, f in x.term_list:
            out = out + c * char_fn(f)
        return out
    return char_fn(x)


def mkw_convolution_func(x, f_fn, g_fn, tree_coproduct, forest_coproduct):
    """Evaluate ``(f * g)(x)`` on an MKW basis element using the paper's
    coproduct.  ``f_fn``, ``g_fn`` are basis-aware character functions
    (callable on trees AND forests, following the :func:`mkw_apply`
    contract).  ``tree_coproduct`` is ``mkw.coproduct_impl``,
    ``forest_coproduct`` is ``mkw.forest_coproduct_impl``.

    The convolution is computed as
    ``sum_{Delta(x)} c * f_fn(left_forest) * g_fn(right_forest)``,
    where Delta is the tree coproduct if ``x`` is a tree and the forest
    coproduct if ``x`` is an ordered forest.  Because we always use the
    paper's coproduct, the resulting convolution is associative (via
    coassociativity of Delta on H_MKW).
    """
    # ForestSum: linearity
    if isinstance(x, ForestSumLike):
        out = 0
        for c, f in x.term_list:
            out = out + c * mkw_convolution_func(
                f, f_fn, g_fn, tree_coproduct, forest_coproduct)
        return out

    if isinstance(x, ForestLike):
        trees = [t for t in x.tree_list if t.list_repr is not None]
        if not trees:
            # Empty forest: convolution = f(empty) * g(empty)
            empty = x.tree_list[0]
            return f_fn(empty) * g_fn(empty)
        if len(trees) == 1:
            # Single-tree forest: reduce to tree-coproduct case
            return mkw_convolution_func(
                trees[0], f_fn, g_fn, tree_coproduct, forest_coproduct)
        cp = forest_coproduct(x)
    else:
        # A Tree/PlanarTree
        cp = tree_coproduct(x)

    out = 0
    for c, left, right in cp.term_list:
        out = out + c * f_fn(left) * g_fn(right)
    return out


def mkw_convolution_power(x, f_fn, exponent,
                          tree_coproduct, forest_coproduct,
                          counit_fn, antipode_fn):
    """MKW convolution power ``f^n`` evaluated at ``x`` (tree or forest).

    ``exponent == 0`` returns the counit.  Positive exponents are computed
    by iterated :func:`mkw_convolution_func`.  Negative exponents use
    the antipode: ``f^{-n}(x) = f^n(S(x))`` evaluated via linearity over
    the ForestSum returned by ``antipode_fn``.
    """
    if exponent == 0:
        return counit_fn(x)
    if exponent == 1:
        return f_fn(x)
    if exponent < 0:
        def inv_fn(y):
            return mkw_convolution_power(
                y, f_fn, -exponent,
                tree_coproduct, forest_coproduct,
                counit_fn, antipode_fn)
        # S(x) is a ForestSum — apply inv_fn linearly over it
        s_x = antipode_fn(x)
        return mkw_apply(s_x, inv_fn)

    # exponent > 1: iterate
    def rec_fn(y):
        return mkw_convolution_power(
            y, f_fn, exponent - 1,
            tree_coproduct, forest_coproduct,
            counit_fn, antipode_fn)
    return mkw_convolution_func(
        x, f_fn, rec_fn, tree_coproduct, forest_coproduct)


def mkw_shuffle_symmetric_value(x, tree_fn):
    """The shuffle-symmetric extension of a tree-only function to an
    ordered forest: ``alpha((t_1, ..., t_k)) = prod alpha(t_i) / k!``.

    This is the correct convention for the CHARACTER α of an exponential
    ``exp(beta . k)`` in a CF method: a single exponential generates a
    shuffle-symmetric character on the basis of ordered forests.  It is
    NOT, however, the correct convention for a convolution result — those
    must be evaluated by recursing through the paper's Delta_forest (see
    :func:`mkw_convolution_func`).

    ``x`` may be a tree (returns ``tree_fn(x)``), a ForestLike (uses the
    Pi/k! formula), or a ForestSum (linear extension).  Tree-only scalar
    maps should wrap themselves with :func:`mkw_base_char_func` rather than
    calling this helper directly.
    """
    if isinstance(x, ForestSumLike):
        out = 0
        for c, f in x.term_list:
            out = out + c * mkw_shuffle_symmetric_value(f, tree_fn)
        return out
    if isinstance(x, ForestLike):
        non_empty = [t for t in x.tree_list if t.list_repr is not None]
        if not non_empty:
            return tree_fn(x.tree_list[0])
        prod = 1
        for t in non_empty:
            prod = prod * tree_fn(t)
        k = len(non_empty)
        if k == 1:
            return prod
        return prod / math.factorial(k)
    return tree_fn(x)


def mkw_base_char_func(tree_fn):
    """Wrap a tree-only scalar function as a basis-aware character for
    the MKW Hopf algebra, with shuffle-symmetric extension to ordered
    forests.  The returned callable accepts any MKW basis element (tree
    or forest) and satisfies the :func:`mkw_apply` contract — so it can
    be fed directly into :func:`mkw_convolution_func`.
    """
    def char(x):
        return mkw_shuffle_symmetric_value(x, tree_fn)
    return char


def _default_mul(a, b):
    return a * b

def func_product(t, func1, func2, coproduct, singleton_reduce=False, product=None, anti1=False):
    # Given the coproduct of some hopf algebra, and two functions func1 and func2,
    # computes the function product evaluated at a tree t, defined by
    # \\mu \\circ (func1 \\otimes func2) \\circ \\Delta (t)
    # where Delta is the coproduct and mu is the algebra multiplication.
    #
    # If singleton_reduce=True, applies singleton_reduced() to the result.
    # This is needed for algebras where the single-node tree is the unit (CEM, GL, PGL).
    #
    # product: optional binary function for combining results. If None, uses
    #   forest juxtaposition (*). For GL/PGL, pass the grafting product.
    # anti1: if True, extend func1 to forests as an anti-homomorphism (reversed order).

    cp = coproduct(t)
    if len(cp) == 0:
        return 0

    _apply1 = anti_forest_apply if anti1 else forest_apply
    _mul = product if product is not None else _default_mul

    out = cp[0][0] * _mul(_apply1(cp[0][1], func1), func2(cp[0][2][0]))
    for c, branches, subtree_ in cp[1:]:
        subtree = subtree_[0]
        out += c * _mul(_apply1(branches, func1), func2(subtree))

    if isinstance(out, (ForestLike, ForestSumLike)):
        if singleton_reduce:
            out = out.singleton_reduced()
        out = out.simplify()

    return out

def func_power(t, func, exponent, coproduct, counit, antipode, singleton_reduce=False, product=None, anti1=False):
    # Given the coproduct, counit and antipode of some hopf algebra,
    # computes the power of func, where the product of functions is
    # defined as above, and f^{-1} = f \\circ antipode.

    if exponent == 0:
        res = counit(t)
    elif exponent == 1:
        res = func(t)
    elif exponent < 0:
        def m(x):
            return func_power(x, func, -exponent, coproduct, counit, antipode, singleton_reduce, product, anti1)
        res = forest_sum_apply(antipode(t), m)
    else:
        def m(x):
            return func_power(x, func, exponent - 1, coproduct, counit, antipode, singleton_reduce, product, anti1)
        res = func_product(t, func, m, coproduct, singleton_reduce, product, anti1)

    if isinstance(res, (ForestLike, ForestSumLike)):
        if singleton_reduce:
            res = res.singleton_reduced()
        res = res.simplify()
    return res
