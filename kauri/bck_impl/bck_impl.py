from functools import cache
import itertools
from ..trees import Tree, Forest, TensorProductSum, EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM, SINGLETON_TREE, SINGLETON_FOREST, SINGLETON_FOREST_SUM
from ..generic_algebra import _forest_apply

def _counit(t):
    return 1 if t.list_repr is None else 0

@cache
def _antipode(t):
    if t.list_repr is None:
        return EMPTY_FOREST_SUM
    elif t.list_repr == tuple():
        return -SINGLETON_FOREST_SUM

    cp = _coproduct(t)
    out = -t.as_forest_sum()
    for c, branches, subtree_ in cp:
        subtree = subtree_[0]
        if subtree._equals(t) or subtree._equals(EMPTY_TREE):
            continue
        out = out - c * _forest_apply(branches, _antipode) * subtree

    return out.reduce()

@cache
def _coproduct_helper(t):
    if t.list_repr is None:
        return [(EMPTY_FOREST, EMPTY_TREE)]
    if t.list_repr == tuple():
        return [(EMPTY_FOREST, SINGLETON_TREE), (SINGLETON_FOREST, EMPTY_TREE)]

    term_list = []
    for rep in t.list_repr:
        subtree = Tree(rep)
        term_list.append(_coproduct_helper(subtree))

    new_term_list = [(Forest((t,)), EMPTY_TREE)]

    for p in itertools.product(*term_list):
        s_repr_ = []
        t_list_ = []
        for f, s in p:
            if s.list_repr is not None:
                s_repr_ += [s.list_repr]
            t_list_ += f.tree_list
        new_term_list.append((Forest(t_list_), Tree(s_repr_)))

    return new_term_list

def _coproduct(t):
    cp = _coproduct_helper(t)
    return TensorProductSum(tuple((1, x[0], x[1]) for x in cp)).reduce()