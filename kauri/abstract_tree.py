"""
Abstract base classes for Trees, Forests and ForestSums
"""

from abc import ABC, abstractmethod
import math
from dataclasses import dataclass
from collections import Counter
from typing import Union

from .utils import (_nodes, _height, _factorial, _sigma,
                    _list_repr_to_level_sequence, _list_repr_to_color_sequence)

######################################
@dataclass(frozen=True)
class AbstractTree(ABC):
######################################
    list_repr: Union[tuple, list, None] = None
    unlabelled_repr = None
    _max_color = 0

    def nodes(self) -> int:
        """
        Returns the number of nodes in a tree, :math:`|t|`

        :return: Number of nodes, :math:`|t|`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.nodes() #Returns 4
        """
        return _nodes(self.unlabelled_repr)

    def colors(self) -> int:
        """
        Returns the number of colors/labels in a labelled tree. Since the labels
        are indexed starting from 0, this is equivalent to one more than the maximum label.

        :return: Number of colors
        :rtype: int

        Example usage::

            Tree([]).colors() # Returns 1
            Tree([0]).colors() # Returns 1
            Tree([[9],1]).colors() # Returns 10
        """
        if self.list_repr is None:
            return 0
        return self._max_color + 1

    def height(self) -> int:
        """
        Returns the height of a tree, given by the number of nodes
        in the longest walk from the root to a leaf.

        :return: Height
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.height() #Returns 3
        """
        return _height(self.unlabelled_repr)

    def factorial(self) -> int:
        """
        Compute the tree factorial, :math:`t!`

        :return: Tree factorial, :math:`t!`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.factorial() #Returns 8
        """
        return _factorial(self.unlabelled_repr)[0]

    def sigma(self) -> int:
        """
        Computes the symmetry factor :math:`\\sigma(t)`, the order of the symmetric
        group of the tree. For a tree :math:`t = [t_1^{m_1} t_2^{m_2} \\cdots t_k^{m_k}]`,
        the symmetry factor satisfies the recursion

        .. math::
            \\sigma(t) = \\prod_{i=1}^k m_i! \\sigma(t_i)^{m_i}.

        :return: Symmetry factor, :math:`\\sigma(t)`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.sigma()
        """
        return _sigma(self.unlabelled_repr)

    def alpha(self) -> int:
        """
        For a tree :math:`t` with :math:`n` nodes, computes the number of
        distinct ways of labelling the nodes of the tree with symbols
        :math:`\\{1, 2, \\ldots, n\\}`, such that:

        - Each vertex receives one and only one label,
        - Labellings that are equivalent under the symmetry group are counted only once,
        - If :math:`(i,j)` is a labelled edge, then :math:`i<j`.

        This number is typically denoted by :math:`\\alpha(t)` and given by

        .. math::
            \\alpha(t) = \\frac{n!}{t! \\sigma(t)}

        :return: :math:`\\alpha(t)`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.alpha()
        """
        return self.beta() // self.factorial()

    def beta(self) -> int:
        """
        For a tree :math:`t` with :math:`n` nodes, computes the number
        of distinct ways of labelling the nodes of the tree with symbols
        :math:`\\{1, 2, \\ldots, n\\}`, such that:

        - Each vertex receives one and only one label,
        - Labellings that are equivalent under the symmetry group are counted only once.

        This number is typically denoted by :math:`\\beta(t)` and given by

        .. math::
            \\beta(t) = \\frac{n!}{\\sigma(t)}

        :return: :math:`\\beta(t)`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.alpha()
        """
        return math.factorial(self.nodes()) // self.sigma()

    def level_sequence(self) -> list:
        """
        Returns the level sequence of the tree, defined as the list
        :math:`{\\ell_1, \\ell_2, \\cdots, \\ell_n}`, where :math:`\\ell_i`
        is the level of the :math:`i^{th}` node when the nodes are ordered lexicographically.

        :return: Level sequence
        :rtype: list

        Example usage::

            t = Tree([[[]],[]])
            t.level_sequence() #Returns [0, 1, 2, 1]
        """
        return _list_repr_to_level_sequence(self.unlabelled_repr)

    def color_sequence(self):#TODO: doc
        return _list_repr_to_color_sequence(self.list_repr)

######################################
@dataclass(frozen=True)
class AbstractForest(ABC):
######################################
    tree_list : Union[tuple, list] = tuple()
    count : Counter = None
    hash_ : int = None

    def _set_counter(self):
        if self.count is None:
            object.__setattr__(self, 'count', Counter(self.simplify().tree_list))

    def _set_hash(self):
        self._set_counter()
        if self.hash_ is None:
            object.__setattr__(self, 'hash_', hash(frozenset(self.count.items())))

    @abstractmethod
    def simplify(self):
        pass

    def nodes(self) -> int:
        """
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns the
        number of nodes in the forest, :math:`\\sum_{i=1}^k |t_i|`.

        :return: Number of nodes
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]])
            f.nodes() #Returns 3
        """
        return sum(t.nodes() for t in self.tree_list)

    def colors(self) -> int:
        """
        Returns the number of colors/labels in the forest. Since the labels are
        indexed starting from 0, this is equivalent to one more than the maximum label.

        :return: Number of colors
        :rtype: int

        Example usage::

            (Tree([[9],0]) * Tree([3])).colors() # Returns 10
        """
        return max(t.colors() for t in self.tree_list)

    def num_trees(self) -> int:
        """
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns the
        number of trees in the forest, :math:`k`.

        :return: Number of trees
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]])
            f.len() #Returns 2
        """
        return len(self.tree_list)

    def factorial(self) -> int:
        """
        Apply the tree factorial to the forest as a multiplicative map.
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns :math:`\\prod_{i=1}^k t_i!`.

        :return: :math:`\\prod_{i=1}^k t_i!`
        :rtype: int

        Example usage::

            f = Tree([[]]) * Tree([[],[]])
            f.factorial() #Returns 6
        """
        return math.prod(x.factorial() for x in self.tree_list)


######################################
@dataclass(frozen=True)
class AbstractForestSum(ABC):
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

    def nodes(self) -> int:
        """
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns the total number of nodes in the forest sum,
        :math:`\\sum_{i=1}^m \\sum_{j=1}^{k_i} |t_{ij}|`.

        :return: Number of nodes
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]]) + 2 * Tree([[],[]])
            f.nodes() #Returns 6
        """
        return sum(f.nodes() for c, f in self.term_list)

    def colors(self) -> int:
        """
        Returns the number of colors/labels in the forest sum. Since the labels are
        indexed starting from 0, this is equivalent to one more than the maximum label.

        :return: Number of colors
        :rtype: int

        Example usage::

            (Tree([[9],0]) * Tree([3]) + Tree([2])).colors() # Returns 10
        """
        return max(f.colors() for _, f in self.term_list)

    def num_trees(self) -> int:
        """
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns the total number of trees in the forest sum, :math:`\\sum_{i=1}^m k_i`.

        :return: Number of trees
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]]) + 2 * Tree([[],[]])
            f.num_trees() #Returns 3
        """
        return sum(f.num_trees() for c, f in self.term_list)

    def num_forests(self) -> int:
        """
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns the total number of forests in the forest sum, :math:`m`.

        :return: Number of forests
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]]) + 2 * Tree([[],[]])
            f.num_trees() #Returns 2
        """
        return len(self.term_list)

    @abstractmethod
    def simplify(self):
        pass

    def factorial(self) -> int:
        """
        Apply the tree factorial to the forest sum as a multiplicative linear map.
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}!`.

        :return: :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}!`
        :rtype: int

        Example usage::

            s = Tree([[],[[]]]) * Tree([]) + Tree([[]])
            s.factorial() #Returns 10
        """
        return sum(c * f.factorial() for c,f in self.term_list)

##############################################
##############################################

def _is_scalar(obj):
    return isinstance(obj, (int, float))

def _is_tree_or_forest(obj):
    return isinstance(obj, (AbstractTree, AbstractForest))

def _is_simplifiable(obj):
    return isinstance(obj, (AbstractForest, AbstractForestSum))

def _is_tree_like(obj):
    return isinstance(obj, (AbstractTree, AbstractForest, AbstractForestSum))
