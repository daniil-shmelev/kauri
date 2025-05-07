"""
Back-end for the BCK module
"""
from functools import cache
import itertools
from ..trees import (Tree, Forest, TensorProductSum,
                     EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM)
from ..generic_algebra import _forest_apply

def _counit(t):
    # Return 1 if t is the empty tree, otherwise 0
    return 1 if t.list_repr is None else 0

@cache
def _antipode(t):
    if t.list_repr is None:
        return EMPTY_FOREST_SUM # Antipode of empty tree is the empty tree
    if t.list_repr == tuple():
        return -t # Antipode of singleton is the negative singleton

    cp = _coproduct(t)
    out = -t.as_forest_sum() # First term, -t
    for c, branches, subtree_ in cp: # Remaining terms
        subtree = subtree_[0] # Convert from Forest to Tree
        if subtree.equals(t) or subtree.equals(EMPTY_TREE):
            continue # We've already included the -t term at the start, so move on
        out = out - c * _forest_apply(branches, _antipode) * subtree

    return out.reduce()

@cache
def _coproduct_helper(t):
    # This returns the coproduct as a list of (Forest, Tree) tuples
    # The function _coproduct then converts this to a tensor product sum
    # and simplifies.

    # We compute the coproduct for a tree t = [t_1, t_2, ..., t_k] recursively.
    # As per https://www2.mathematik.hu-berlin.de/~kreimer/wp-content/uploads/Foissy.pdf,
    # the coproduct can be written as a sum over admissible cuts, defined as cuts
    # where every walk from the root to a leaf contains at most one cut edge.
    # The coproduct is then a sum of tensor products, where each term is the
    # product of the forest of branches resulting from a cut and the remaining tree
    # connected to the root. Denote these by P_c(t) and R_c(t) respectively, for a
    # cut c.

    # The recursion works on the idea that P_c(t) is the union of P_c(t_i),
    # and R_c(t) = [R_c(t_1), R_c(t_2), ..., R_c(t_k)]. This allows us to use the
    # coproducts of t_i to compute the coproduct of t.

    # Caching of this function makes it fairly efficient for large computations.

    if t.list_repr is None: # Empty tree
        return [(EMPTY_FOREST, EMPTY_TREE)]
    if t.list_repr == tuple(): # Singleton tree
        return [(EMPTY_FOREST, t), (t.as_forest(), EMPTY_TREE)]

    # Compute the coproducts of t_1, t_2, ..., t_k
    subtree_coproducts = []
    for rep in t.list_repr[:-1]: # Recall last element is the root label, so take [:-1]
        subtree = Tree(rep)
        subtree_coproducts.append(_coproduct_helper(subtree))

    t_coproduct = [(Forest((t,)), EMPTY_TREE)] # First term of coproduct, t x empty tree

    # Remaining terms, compute these recursively
    for p in itertools.product(*subtree_coproducts):
        R_c_repr = []
        P_c_tree_list = []
        for f, s in p:
            if s.list_repr is not None:
                R_c_repr += [s.list_repr]
            P_c_tree_list += f.tree_list
        R_c_repr += [t.list_repr[-1]] #Add label of root
        t_coproduct.append((Forest(P_c_tree_list), Tree(R_c_repr)))

    return t_coproduct

def _coproduct(t):
    cp = _coproduct_helper(t)
    return TensorProductSum(tuple((1, x[0], x[1]) for x in cp)).reduce()
