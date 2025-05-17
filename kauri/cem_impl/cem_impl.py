"""
Back-end for the CEM module
"""
from functools import cache
import itertools
from ..trees import Tree, Forest
from ..tensor_product import TensorProductSum

# We adopt the singleton-reduced coproduct, which defines a Hopf algebra
# on planar trees quotiented by ([] - 1). As such, characters on the
# resulting Hopf algebra must satisfy \phi([]) = 1

def _counit(t):
    # Return 1 if t is the tree with one node, otherwise 0
    return 1 if len(t.list_repr) == 1 else 0

@cache
def _antipode(t):

    # Consider the empty tree and the single node tree to be equal, since the latter is the unit
    if t.list_repr is None:
        return Tree([]).as_forest_sum()
    if len(t.list_repr) == 1:
        return t.as_forest_sum()

    cp = _coproduct(t)
    out = -t.as_forest_sum() # First term, -t
    for c, branches, subtree_ in cp: # Remaining terms
        subtree = subtree_[0] # Convert from Forest to Tree
        if branches.equals(t.as_forest()) or subtree.equals(t):
            continue # We've already included the -t term at the start, so move on
        out = out - c * _antipode(subtree) * branches

    return out.singleton_reduced().simplify() # Single node tree is the unit, so can apply .singleton_reduced() here

@cache
def _coproduct_helper(t):
    # This returns the coproduct as a list of Forests and a list of Trees
    # The function _coproduct then converts this to a tensor product sum
    # and simplifies.

    # We compute the coproduct for a tree t = [t_1, t_2, ..., t_k] recursively,
    # similarly to how we did for the BCK coproduct. Instead of cutting edges,
    # however, we are now contracting them.

    # Caching of this function makes it fairly efficient for large computations.

    if t.list_repr is None: # Empty tree
        return [Tree([]).as_forest()], [Tree([])]
    if len(t.list_repr) == 1: # Singleton tree
        return [t.as_forest()], [t]

    # Compute the coproducts of t_1, t_2, ..., t_k
    subtree_coproduct_trees = []
    subtree_coproduct_forests = []
    for rep in t.list_repr[:-1]: # Recall last element is the root label, so take [:-1]
        b, s = _coproduct_helper(Tree(rep))
        subtree_coproduct_trees.append(s)
        subtree_coproduct_forests.append(b)

    # Now compute the coproduct of t
    t_coproduct_trees = []
    t_coproduct_forests = []

    k = len(subtree_coproduct_trees) # Number of subtrees

    # For each edge connecting a subtree to the root, we can either contract
    # it or not. The "edges" parameter here is a list of binary flags which
    # determine which edges to contract. We must loop over all possible
    # combinations of these flags, of which there are 2^k.
    for edges in itertools.product([0, 1], repeat=k):

        # For a given choice of edges to contract, we must now loop over all
        # possible combinations of coproduct terms of subtrees. This is the same
        # logic as in the BCK case, but split into two loops handling the trees
        # and forests separately, as its slightly easier to accommodate the
        # contraction logic this way.

        # Get trees
        for p in itertools.product(*subtree_coproduct_trees):
            rep = []
            for edge, t_ in zip(edges, p):
                if t_.list_repr is None:
                    continue
                rep += t_.list_repr[:-1] if edge else [t_.list_repr] # Add to tree, contracting the edge if necessary
            t_coproduct_trees.append(Tree(rep))

        # Get forests
        for p in itertools.product(*subtree_coproduct_forests):
            # Must ensure that the first tree in the forest is connected to the root,
            # as it's important to know what this tree is for the recursion.
            # If no such tree, add an empty tree to the forest to signify this
            # Forest constructor does not call Forest.simplify(), meaning this empty tree will survive
            t_list_ = []
            root_tree_repr = [] # The tree connected to the root
            for edge, f in zip(edges, p):
                if edge:
                    root_tree_repr += [f.tree_list[0].list_repr]
                    t_list_ += f.tree_list[1:]
                else:
                    t_list_ += f.tree_list
            t_list_ = [Tree(root_tree_repr)] + t_list_ # Keep tree connected to root at the start
            t_coproduct_forests.append(Forest(t_list_))

    return t_coproduct_forests, t_coproduct_trees

def _coproduct(t):
    f, s = _coproduct_helper(t)
    cp = zip([x.simplify().singleton_reduced() for x in f], s)
    return TensorProductSum(tuple((1, x[0], x[1]) for x in cp)).simplify()
