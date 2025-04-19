from functools import cache
import itertools
from kauri.trees import Tree, Forest, ForestSum, EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM, ZERO_FOREST_SUM, SINGLETON_TREE, SINGLETON_FOREST, SINGLETON_FOREST_SUM
from kauri.generic_algebra import _forest_apply
from kauri.tensor_product import TensorSum

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
    for subtree, branches in cp:
        if subtree._equals(t) or subtree._equals(EMPTY_TREE):
            continue
        out = out - _forest_apply(branches, _antipode) * subtree

    return out.reduce()

@cache
def _coproduct(t):
    if t.list_repr is None:
        return [(EMPTY_TREE, EMPTY_FOREST)]
    if t.list_repr == tuple():
        return [(SINGLETON_TREE, EMPTY_FOREST), (EMPTY_TREE, SINGLETON_FOREST)]

    term_list = []
    for rep in t.list_repr:
        subtree = Tree(rep)
        term_list.append(_coproduct(subtree))

    new_term_list = [(EMPTY_TREE, Forest((t,)))]

    for p in itertools.product(*term_list):
        s_repr_ = []
        t_list_ = []
        for s, f in p:
            if s.list_repr is not None:
                s_repr_ += [s.list_repr]
            t_list_ += f.tree_list
        new_term_list.append((Tree(s_repr_),Forest(t_list_)))

    return TensorSum(new_term_list)