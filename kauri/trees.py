"""
Tree, Forest, ForestSum and TensorProductSum classes.

The classes `Tree`, `Forest` and `ForestSum` are immutable and hashable.
The hash is generated in such a way that two elements of the same class which are equivalent
(e.g. two different orderings of the same tree) will have the same hash.
However, this is not the case across classes. For example, for a Tree t, `hash(t)`,
`hash(t.as_forest())` and `hash(t.as_forest_sum())` are different.

The class `Tree` is totally ordered by the lexicographic ordering. If the trees have
the same structure but are colored differently, they are ordered based on color, with
the color of the highest levels of the trees being the primary ordering.

"""

from dataclasses import dataclass
from functools import total_ordering
from collections import Counter
from typing import Union
import warnings

from .utils import (_sorted_list_repr, _to_list, _next_layout, _level_sequence_to_list_repr,
                    _check_valid, _to_labelled_tuple, _get_max_color, _to_unlabelled_tuple,
                    LabelledReprComparison)

from .abstract_tree import AbstractTree, AbstractForest, AbstractForestSum, TensorProductSum

######################################
@dataclass(frozen=True)
@total_ordering
class Tree(AbstractTree):
    """
    A single non-planar (un)labelled rooted tree, initialised by its list representation.
    For example, the unlabelled cherry tree has the list representation [[],[]]. Noting
    that every list corresponds to a node, we can apply a labelling/coloring by setting the last
    element of the list to be a non-negative integer. For example, [[2], [1], 0] corresponds
    to the cherry tree, with the root node labelled by 0, the left leaf labelled by 2 and
    the right leaf labelled by 1. If a label is left out, it will default to 0.

    :param list_repr: The nested list representation of the tree

    Example usage::

            t1 = Tree([[[]],[]]) # An unlabelled tree
            t2 = Tree([[[3],1],[2],0]) # A labelled tree
            t3 = Tree([[[3],1],[2]]) # This is the same as t2, since the missing label defaults to 0
    """
######################################
    list_repr: Union[tuple, list, None] = None
    unlabelled_repr = None
    _max_color = 0

    def __repr__(self):
        return super().__repr__()

    def __hash__(self):
        return hash(self.sorted_list_repr())

    def __lt__(self, other):
        # Deal with empty trees
        if self.list_repr is None:
            if other.list_repr is None:
                return False
            return True
        if other.list_repr is None:
            return False

        # If trees are non-empty
        if self.nodes() != other.nodes():
            return self.nodes() < other.nodes()
        return LabelledReprComparison(self.sorted_list_repr()) < LabelledReprComparison(other.sorted_list_repr())

    def sorted_list_repr(self) -> list:
        """
        Returns the list representation of the sorted tree,
        where the heaviest branches are rotated to the left.

        :return: Sorted list representation
        :rtype: list

        Example usage::

            t = Tree([[],[[]]])
            t.sorted_list_repr() #Returns [[[]],[]]
        """
        return _sorted_list_repr(self.list_repr)

    def sorted(self) -> 'Tree':
        """
        Returns the sorted tree, where the heaviest branches are rotated to the left.

        :return: Sorted tree
        :rtype: Tree

        Example usage::

            t = Tree([[],[[]]])
            t.level_sequence() #Returns Tree([[[]],[]])
        """
        return Tree(self.sorted_list_repr())

    def equals(self, other_tree):
        return self.sorted_list_repr() == other_tree.sorted_list_repr()

    def __eq__(self, other_tree):
        return super().__eq__(other_tree)

    def __next__(self) -> 'Tree':
        """
        Generates the next tree with respect to the lexicographic order.
        If the tree is labelled, the labelling will be ignored.

        :return: Next tree
        :rtype: Tree

        Example usage::

                t = Tree([[],[]])
                next(t) # returns Tree([[[[]]]])
        """
        if self._max_color > 0:
            warnings.warn("Calling next() on a labelled tree will ignore the labelling.")
        if self.list_repr is None:
            return Tree([])

        layout = self.level_sequence()
        next_ = _next_layout(layout)
        return Tree(_level_sequence_to_list_repr(next_))

######################################
@dataclass(frozen=True)
class Forest(AbstractForest):
    """
    A commutative product of trees.

    :param tree_list: A list of trees contained in the forest

    Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])

            f = Forest([t1,t2,t3])
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
class ForestSum(AbstractForestSum):
    """
    A linear combination of forests.

    :param term_list: A list or tuple containing tuples of coefficients and
        forests representing terms of the sum. If a term contains a tree, it
        will be converted to a forest on initialisation.

    Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])

            s = ForestSum([(1, t1), (-2, t1*t2), (1, t2*t3)])
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

Tree._tree_class = Tree
Tree._forest_class = Forest
Tree._forest_sum_class = ForestSum

Forest._tree_class = Tree
Forest._forest_class = Forest
Forest._forest_sum_class = ForestSum

ForestSum._tree_class = Tree
ForestSum._forest_class = Forest
ForestSum._forest_sum_class = ForestSum

##############################################
##############################################

EMPTY_TREE = Tree(None)
EMPTY_FOREST = Forest((EMPTY_TREE,))
EMPTY_FOREST_SUM = ForestSum( ( (1, EMPTY_FOREST), ) )
ZERO_FOREST_SUM = ForestSum( ( (0, EMPTY_FOREST), ) )
