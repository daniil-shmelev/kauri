from functools import cache
import itertools
from ..trees import Tree, Forest, TensorProductSum, EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM, SINGLETON_TREE, SINGLETON_FOREST, SINGLETON_FOREST_SUM, ZERO_FOREST_SUM
from ..generic_algebra import _forest_apply

def _counit(t):
    return 1 if t.list_repr == tuple() else 0

@cache
def _antipode(t):
    if t.list_repr is None or t.list_repr == tuple(): #Consider the empty tree and the single node tree to be equal, since the latter is the unit
        return SINGLETON_FOREST_SUM

    cp = _coproduct(t)
    out = -t.as_forest_sum()
    for c, subtree_, branches in cp:
        subtree = subtree_[0]
        if branches._equals(t.as_forest()) or subtree._equals(t):
            continue
        out = out - c * _antipode(subtree) * branches

    return out.singleton_reduced().reduce()

@cache
def _coproduct_helper(t):
    if t.list_repr is None or t.list_repr == tuple():
        return [SINGLETON_TREE], [SINGLETON_FOREST]

    tree_list = []
    forest_list = []
    for rep in t.list_repr:
        subtree = Tree(rep)
        s, b = _coproduct_helper(subtree)
        tree_list.append(s)
        forest_list.append(b)

    new_tree_list = []
    new_forest_list = []

    num_branches = len(tree_list)

    for edges in itertools.product([0, 1], repeat=num_branches):

        for p in itertools.product(*tree_list):
            rep = []
            for i, t in enumerate(p):
                if t.list_repr is None:
                    continue
                if edges[i]:
                    rep += t.list_repr
                else:
                    rep += [t.list_repr]
            new_tree_list.append(Tree(rep))

        for p in itertools.product(*forest_list):
            # Must ensure that the first tree in the forest is connected to the root
            # If no such tree, add an empty tree to the forest to signify this
            # Forest constructor does not call Forest.reduce(), meaning this empty tree will survive
            t_list_ = []
            root_tree_repr = []
            for i, f in enumerate(p):
                if edges[i]:
                    root_tree_repr += [f.tree_list[0].list_repr]
                    t_list_ += f.tree_list[1:]
                else:
                    t_list_ += f.tree_list
            t_list_ = [Tree(root_tree_repr)] + t_list_
            new_forest_list.append(Forest(t_list_).singleton_reduced())

    return new_tree_list, new_forest_list

def _coproduct(t):
    s, f = _coproduct_helper(t)
    cp = zip(s, [x.reduce() for x in f])
    return TensorProductSum(tuple((1, x[0], x[1]) for x in cp))