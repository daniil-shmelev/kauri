"""
Utility functions for dealing with generic Hopf algebras on trees
"""
from .trees import Forest, ForestSum, _is_simplifiable

def _forest_apply(f, func):
    # Apply a function func multiplicatively to a forest f
    out = 1
    for t in f.tree_list:
        out = out * func(t)

    if _is_simplifiable(out):
        out = out.simplify()
    return out

def _forest_sum_apply(fs, func):
    # Applies a function func linearly and multiplicatively to a forest sum fs
    out = 0
    for c, f in fs.term_list:
        term = 1
        for t in f.tree_list:
            term = term * func(t)
        out += c * term

    if _is_simplifiable(out):
        out = out.simplify()
    return out

def _apply(t, func):
    # Applies a function func as a linear multiplicative map to a Forest or ForestSum t
    if isinstance(t, Forest):
        return _forest_apply(t, func)
    if isinstance(t, ForestSum):
        return _forest_sum_apply(t, func)
    return func(t)

def _func_product(t, func1, func2, coproduct):
    # Given the coproduct of some hopf algebra, and two functions func1 and func2,
    # computes the function product evaluated at a tree t, defined by
    # \\mu \\circ (func1 \\otimes func2) \\circ \\Delta (t)
    # where Delta is the coproduct and mu is defined as the commutative
    # juxtaposition of trees.

    cp = coproduct(t)
    # a(branches) * b(subtrees)
    if len(cp) == 0:
        return 0
    out = cp[0][0] * _forest_apply(cp[0][1], func1) * func2(cp[0][2][0]) # cp[0][2] is a forest with one tree, which is cp[0][2][0]
    for c, branches, subtree_ in cp[1:]:
        subtree = subtree_[0] # subtree_ is a forest with one tree, which is subtree_[0]
        out += c * _forest_apply(branches, func1) * func2(subtree)

    if _is_simplifiable(out):
        out = out.simplify()

    return out

def _func_power(t, func, exponent, coproduct, counit, antipode):
    # Given the coproduct, counit and antipode of some hopf algebra,
    # computes the power of func, where the product of functions is
    # defined as above, and f^{-1} = f \\circ antipode.

    if exponent == 0:
        res = counit(t)
    elif exponent == 1:
        res = func(t)
    elif exponent < 0:
        def m(x):
            return _func_power(x, func, -exponent, coproduct, counit, antipode)
        res = _forest_sum_apply(antipode(t), m)
    else:
        def m(x):
            return _func_power(x, func, exponent - 1, coproduct, counit, antipode)
        res = _func_product(t, func, m, coproduct)

    if _is_simplifiable(res):
        res = res.simplify()
    return res
