from functools import cache
import itertools
from ..trees import Tree, Forest, TensorProductSum, EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM, SINGLETON_TREE, SINGLETON_FOREST, SINGLETON_FOREST_SUM, ZERO_FOREST_SUM
from ..generic_algebra import _forest_apply

#We adopt the singleton-reduced coproduct, which defines a Hopf algebra on planar trees quotiented by ([] - 1).
# As such, characters on the resulting Hopf algebra must satisfy \phi([]) = 1

def _counit(t):
    return 1 if t.list_repr == tuple() else 0

@cache
def _antipode(t):
    if t.list_repr is None or t.list_repr == tuple(): #Consider the empty tree and the single node tree to be equal, since the latter is the unit
        return SINGLETON_FOREST_SUM

    cp = _coproduct(t)
    out = -t.as_forest_sum()
    for c, branches, subtree_ in cp:
        subtree = subtree_[0]
        if branches._equals(t.as_forest()) or subtree._equals(t):
            continue
        out = out - c * _antipode(subtree) * branches

    return out.singleton_reduced().reduce()

@cache
def _coproduct_helper(t):
    if t.list_repr is None:
        raise
    if t.list_repr == tuple():
        return [SINGLETON_FOREST], [SINGLETON_TREE]

    tree_list = []
    forest_list = []
    for rep in t.list_repr:
        subtree = Tree(rep)
        b, s = _coproduct_helper(subtree)
        tree_list.append(s)
        forest_list.append(b)

    new_tree_list = []
    new_forest_list = []

    num_branches = len(tree_list)

    for edges in itertools.product([0, 1], repeat=num_branches):

        for p in itertools.product(*tree_list):
            rep = []
            for i, t_ in enumerate(p):
                if t_.list_repr is None:
                    continue
                if edges[i]:
                    rep += t_.list_repr
                else:
                    rep += [t_.list_repr]
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
            new_forest_list.append(Forest(t_list_))

    return new_forest_list, new_tree_list

def _coproduct(t):
    f, s = _coproduct_helper(t)
    cp = zip([x.reduce().singleton_reduced() for x in f], s)
    return TensorProductSum(tuple((1, x[0], x[1]) for x in cp)).reduce()