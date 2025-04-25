from .trees import Tree, Forest, ForestSum, _is_reducible

def _forest_apply(f, func):
    out = 1
    for t in f.tree_list:
        out = out * func(t)

    if _is_reducible(out):
        out = out.reduce()
    return out

def _forest_sum_apply(fs, func):
    out = 0
    for c, f in fs.term_list:
        term = 1
        for t in f.tree_list:
            term = term * func(t)
        out += c * term

    if _is_reducible(out):
        out = out.reduce()
    return out

def _apply(t, func):
    if isinstance(t, Forest):
        return _forest_apply(t, func)
    if isinstance(t, ForestSum):
        return _forest_sum_apply(t, func)
    else:
        return func(t)

def _func_product(t, func1, func2, coproduct):
    cp = coproduct(t)
    # a(branches) * b(subtrees)
    if len(cp) == 0:
        return 0
    out = cp[0][0] * _forest_apply(cp[0][1], func1) * func2(cp[0][2][0])
    for c, branches, subtree_ in cp[1:]:
        subtree = subtree_[0]
        out += c * _forest_apply(branches, func1) * func2(subtree)

    if _is_reducible(out):
        out = out.reduce()

    return out

def _func_power(t, func, exponent, coproduct, counit, antipode):
    res = None
    if exponent == 0:
        res = counit(t)
    elif exponent == 1:
        res = func(t)
    elif exponent < 0:
        m = lambda x : _func_power(x, func, -exponent, coproduct, counit, antipode)
        res = _forest_sum_apply(antipode(t), m)
    else:
        res = _func_product(t, func, lambda x : _func_power(x, func, exponent - 1, coproduct, counit, antipode), coproduct)

    if _is_reducible(res):
        res = res.reduce()
    return res