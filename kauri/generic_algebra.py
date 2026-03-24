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
Utility functions for dealing with generic Hopf algebras on trees
"""
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

def func_product(t, func1, func2, coproduct):
    # Given the coproduct of some hopf algebra, and two functions func1 and func2,
    # computes the function product evaluated at a tree t, defined by
    # \\mu \\circ (func1 \\otimes func2) \\circ \\Delta (t)
    # where Delta is the coproduct and mu is defined as the commutative
    # juxtaposition of trees.

    cp = coproduct(t)
    # a(branches) * b(subtrees)
    if len(cp) == 0:
        return 0
    out = cp[0][0] * forest_apply(cp[0][1], func1) * func2(cp[0][2][0]) # cp[0][2] is a forest with one tree, which is cp[0][2][0]
    for c, branches, subtree_ in cp[1:]:
        subtree = subtree_[0] # subtree_ is a forest with one tree, which is subtree_[0]
        out += c * forest_apply(branches, func1) * func2(subtree)

    if isinstance(out, (ForestLike, ForestSumLike)):
        out = out.simplify()

    return out

def func_power(t, func, exponent, coproduct, counit, antipode):
    # Given the coproduct, counit and antipode of some hopf algebra,
    # computes the power of func, where the product of functions is
    # defined as above, and f^{-1} = f \\circ antipode.

    if exponent == 0:
        res = counit(t)
    elif exponent == 1:
        res = func(t)
    elif exponent < 0:
        def m(x):
            return func_power(x, func, -exponent, coproduct, counit, antipode)
        res = forest_sum_apply(antipode(t), m)
    else:
        def m(x):
            return func_power(x, func, exponent - 1, coproduct, counit, antipode)
        res = func_product(t, func, m, coproduct)

    if isinstance(res, (ForestLike, ForestSumLike)):
        res = res.simplify()
    return res
