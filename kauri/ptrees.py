"""
PTree, PForest, PForestSum and TensorProductSum classes.

The classes `PTree`, `PForest` and `PForestSum` are immutable and hashable.
The hash is generated in such a way that two elements of the same class which are equivalent
(e.g. two different orderings of the same tree) will have the same hash.
However, this is not the case across classes. For example, for a PTree t, `hash(t)`,
`hash(t.as_forest())` and `hash(t.as_forest_sum())` are different.

The class `PTree` is totally ordered by the lexicographic ordering. If the trees have
the same structure but are colored differently, they are ordered based on color, with
the color of the highest levels of the trees being the primary ordering.

"""#TODO: Docs

from dataclasses import dataclass
from collections import Counter
from typing import Union

from .abstract_tree import AbstractTree, AbstractForest, AbstractForestSum, TensorProductSum

######################################
@dataclass(frozen=True)
class PTree(AbstractTree):
    """
    A single non-planar (un)labelled rooted tree, initialised by its list representation.
    For example, the unlabelled cherry tree has the list representation [[],[]]. Noting
    that every list corresponds to a node, we can apply a labelling/coloring by setting the last
    element of the list to be a non-negative integer. For example, [[2], [1], 0] corresponds
    to the cherry tree, with the root node labelled by 0, the left leaf labelled by 2 and
    the right leaf labelled by 1. If a label is left out, it will default to 0.

    :param list_repr: The nested list representation of the tree

    Example usage::

            t1 = PTree([[[]],[]]) # An unlabelled tree
            t2 = PTree([[[3],1],[2],0]) # A labelled tree
            t3 = PTree([[[3],1],[2]]) # This is the same as t2, since the missing label defaults to 0
    """
######################################
    list_repr: Union[tuple, list, None] = None
    unlabelled_repr = None
    _max_color = 0

    def __repr__(self):
        return super().__repr__()

    def __hash__(self):
        return super().__hash__()

    def equals(self, other_tree):
        return self.list_repr == other_tree.list_repr

    def __eq__(self, other_tree):
        return super().__eq__(other_tree)

######################################
@dataclass(frozen=True)
class PForest(AbstractForest):
    """
    A commutative product of trees.

    :param tree_list: A list of trees contained in the forest

    Example usage::

            t1 = PTree([])
            t2 = PTree([[]])
            t3 = PTree([[[]],[]])

            f = PForest([t1,t2,t3])
    """
######################################
    tree_list : Union[tuple, list] = tuple()
    count : Counter = None
    hash_ : int = None

    def __repr__(self):
        return super().__repr__()

    def __hash__(self):
        self._set_hash()
        return self.hash_

    def _set_counter(self):
        if self.count is None:
            object.__setattr__(self, 'count', Counter(self.simplify().tree_list))

    def _set_hash(self):
        self._set_counter()
        if self.hash_ is None:
            object.__setattr__(self, 'hash_', hash(frozenset(self.count.items())))

    def equals(self, other_forest):
        self._set_counter()
        other_forest._set_counter()
        return self.count == other_forest.count

    def __eq__(self, other):
        return super().__eq__(other)

######################################
@dataclass(frozen=True)
class PForestSum(AbstractForestSum):
    """
    A linear combination of forests.

    :param term_list: A list or tuple containing tuples of coefficients and
        forests representing terms of the sum. If a term contains a tree, it
        will be converted to a forest on initialisation.

    Example usage::

            t1 = PTree([])
            t2 = PTree([[]])
            t3 = PTree([[[]],[]])

            s = PForestSum([(1, t1), (-2, t1*t2), (1, t2*t3)])
            s == t1 - 2 * t1 * t2 + t2 * t3 #True
    """
######################################
    term_list : Union[tuple, list] = tuple()
    count : Counter = None
    hash_ : int = None

    def _set_counter(self):
        if self.count is None:
            object.__setattr__(self, 'count', Counter(self.simplify().term_list))

    def _set_hash(self):
        self._set_counter()
        if self.hash_ is None:
            object.__setattr__(self, 'hash_', hash(frozenset(self.count.items())))

    def __hash__(self):
        self._set_hash()
        return self.hash_

    def __repr__(self):
        return super().__repr__()

    def equals(self, other):
        self._set_counter()
        other._set_counter()
        return self.count == other.count

    def __eq__(self, other):
        return super().__eq__(other)


##############################################
##############################################

PTree._tree_class = PTree
PTree._forest_class = PForest
PTree._forest_sum_class = PForestSum

PForest._tree_class = PTree
PForest._forest_class = PForest
PForest._forest_sum_class = PForestSum

PForestSum._tree_class = PTree
PForestSum._forest_class = PForest
PForestSum._forest_sum_class = PForestSum

##############################################
##############################################

EMPTY_PTREE = PTree(None)
EMPTY_PFOREST = PForest((EMPTY_PTREE,))
EMPTY_PFOREST_SUM = PForestSum( ( (1, EMPTY_PFOREST), ) )
ZERO_PFOREST_SUM = PForestSum( ( (0, EMPTY_PFOREST), ) )
