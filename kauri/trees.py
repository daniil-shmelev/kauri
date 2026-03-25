# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

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

import math
import numbers
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import total_ordering
from typing import Union
import warnings

import sympy

from .utils import (_nodes, _height, _factorial, _sigma,
                    _sorted_list_repr, _list_repr_to_level_sequence,
                    _to_list, _next_layout, _next_planar_layout, _level_sequence_to_list_repr,
                    _check_valid, _to_labelled_tuple, _get_max_color, _to_unlabelled_tuple,
                    _list_repr_to_color_sequence, LabelledReprComparison)
from ._protocols import ForestLike, TreeLike


def _frozen_copy(self):
    return self


def _frozen_deepcopy(self, memodict=None):
    if memodict is None:
        memodict = {}
    memodict[id(self)] = self
    return self


def _lazy_count(self, items_attr):
    if self.count is None:
        object.__setattr__(self, 'count', Counter(getattr(self.simplify(), items_attr)))


def _lazy_hash(self, items_attr):
    _lazy_count(self, items_attr)
    if self.hash_ is None:
        object.__setattr__(self, 'hash_', hash(frozenset(self.count.items())))
    return self.hash_

######################################
@dataclass(frozen=True)
@total_ordering
class Tree:
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

    def __post_init__(self):
        if self.list_repr is not None:
            if not _check_valid(self.list_repr):
                raise ValueError(repr(self.list_repr) + " is not a valid list representation for a tree.")
            tuple_repr = _to_labelled_tuple(self.list_repr)
            object.__setattr__(self, 'list_repr', tuple_repr)
            unlabelled_repr = _to_unlabelled_tuple(tuple_repr)
            object.__setattr__(self, 'unlabelled_repr', unlabelled_repr)
            object.__setattr__(self, '_max_color', _get_max_color(tuple_repr))

    __copy__ = _frozen_copy
    __deepcopy__ = _frozen_deepcopy

    def __repr__(self):
        if self.list_repr is None:
            return "\u2205"
        if self._max_color == 0:
            return repr(_to_list(self.unlabelled_repr))
        return repr(_to_list(self.list_repr))

    def _repr_svg_(self):
        from .display import _to_svg
        return _to_svg(self)

    def __hash__(self):
        return hash(self.sorted_list_repr())

    def unjoin(self) -> 'Forest':
        """
        For a tree :math:`t = [t_1, t_2, ..., t_k]`, returns the forest :math:`t_1 t_2 \\cdots t_k`.
        In :cite:`connes1999hopf`, this map is denoted by :math:`B_-`.

        :return: :math:`t_1 t_2 \\cdots t_k`
        :rtype: Forest

        Example usage::

            t = Tree([[[]],[]])
            t.unjoin() #Returns Tree([[]]) * Tree([])
        """
        if self.list_repr is None:
            return EMPTY_FOREST
        return Forest(tuple(Tree(rep) for rep in self.list_repr[:-1]))

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

    def density(self) -> float:
        """
        Density of the tree, :math:`\\gamma(t) = t! / |t|!`.

        :return: Density, :math:`\\gamma(t)`
        :rtype: float

        Example usage::

            t = Tree([[[]],[]])
            t.density() #Returns 1/3
        """
        return self.factorial() / math.factorial(self.nodes())

    def sign(self) -> 'ForestSum':
        """
        Returns the tree signed by the number of nodes, :math:`(-1)^{|t|} t`.

        :return: Signed tree, :math:`(-1)^{|t|} t`
        :rtype: ForestSum

        Example usage::

            t = Tree([[[]],[]])
            t.sign()
        """
        return self.as_forest_sum() if self.nodes() % 2 == 0 else -self

    def __mul__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> Union['Forest', 'ForestSum']:
        """
        Multiplies a tree by a:

        - scalar, returning a ForestSum
        - Tree, returning a Forest,
        - Forest, returning a Forest,
        - ForestSum, returning a ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum

        Example usage::

            t = 2 * Tree([[]]) * Forest([Tree([]), Tree([[],[]])])
        """
        if _is_scalar(other):
            out = ForestSum(( (other,self),  ))
        elif isinstance(other, Tree):
            out = Forest((self, other))
        elif isinstance(other, Forest):
            out = Forest((self,) + other.tree_list)
        elif isinstance(other, ForestSum):
            out = ForestSum(tuple((c, self * f) for c,f in other.term_list))
        else:
            _check_compatible(self, other)
            raise TypeError("Cannot multiply Tree by object of type " + str(type(other)))

        return out.simplify()

    __rmul__ = __mul__

    def __pow__(self, n : int) -> 'Forest':
        """
        Returns the :math:`n^{th}` power of a tree for a positive integer
        :math:`n`, given by a forest with :math:`n` copies of the tree.

        :param n: Exponent, a positive integer

        Example usage::

            t = Tree([[]]) ** 3
        """
        if not isinstance(n, int):
            raise TypeError("Exponent in Tree.__pow__ must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise Tree to a negative power")
        if n == 0:
            return EMPTY_FOREST

        out = Forest((self,) * n)
        return out.simplify()

    def __add__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'ForestSum':
        """
        Adds a tree to a scalar, Tree, Forest or ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 + Tree([[]]) + Forest([Tree([]), Tree([[],[]])])
        """
        if _is_scalar(other):
            out = ForestSum((  (1, self), (other, EMPTY_FOREST)  ))
        elif isinstance(other, (Tree, Forest)):
            out = ForestSum((  (1, self), (1, other)  ))
        elif isinstance(other, ForestSum):
            _check_compatible(self, other)
            out = ForestSum( ((1, self),) + other.term_list )
        else:
            _check_compatible(self, other)
            raise TypeError("Cannot add Tree and " + str(type(other)))

        return out.simplify()

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    __radd__ = __add__

    def __neg__(self):
        return self * (-1)

    def __eq__(self, other : Union['Tree', 'Forest', 'ForestSum']) -> bool:
        """
        Compares the tree with another object and returns true if they represent
        the same tree, regardless of class type (Tree, Forest or ForestSum) or
        possible reorderings of the same tree.

        :param other: Tree, Forest or ForestSum
        :rtype: bool

        Example usage::

            Tree([[],[]]) == Tree([[],[]]).as_forest() #True
            Tree([[],[]]) == Tree([[],[]]).as_forest_sum() #True
            Tree([[[]],[]]) == Tree([[],[[]]]) #True
        """
        if _is_scalar(other):
            return self.as_forest_sum() == other * EMPTY_TREE
        if isinstance(other, Tree):
            return self.equals(other)
        if isinstance(other, Forest):
            return self.as_forest() == other
        if isinstance(other, ForestSum):
            return self.as_forest_sum() == other
        return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, Tree):
            return NotImplemented
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

    def as_forest(self) -> 'Forest':
        """
        Returns the tree t as a forest. Equivalent to Forest([t]).

        :return: Tree as a forest
        :rtype: Forest

        Example usage::

            t = Tree([[],[[]]])
            t.as_forest() #Returns Forest([Tree([[[]],[]])])
        """
        return Forest((self,))

    def as_forest_sum(self) -> 'ForestSum':
        """
        Returns the tree t as a forest sum. Equivalent to ForestSum([Forest([t])]).

        :return: Tree as a forest sum
        :rtype: ForestSum

        Example usage::

            t = Tree([[],[[]]])
            t.as_forest_sum() #Returns ForestSum([Forest([Tree([[[]],[]])])])
        """
        return ForestSum(( (1, self), ))

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

    def __matmul__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'TensorProductSum':
        """
        Returns the tensor product of a Tree and a scalar, Tree, Forest or ForestSum.

        :param other: Other
        :type other: int | float | Tree | Forest | ForestSum
        :return: Tensor product
        :rtype: TensorProductSum

        Example usage::

            Tree([]) @ (Tree([[]]) + Tree([]) * Tree([[],[]])) # Returns 1 [] ⊗ [[]]+1 [] ⊗ [] [[], []]
        """
        if _is_scalar(other):
            return TensorProductSum(( (other, self.as_forest(), EMPTY_FOREST), ))
        if isinstance(other, (Tree, Forest)):
            return TensorProductSum(( (1, self.as_forest(), other.as_forest()), ))
        if isinstance(other, ForestSum):
            term_list = []
            for c, f in other:
                term_list.append((c, self, f))
            return TensorProductSum(term_list)
        raise TypeError("Cannot take tensor product of Tree and " + str(type(other)))

    def unlabelled(self):
        """
        Returns the unlabelled version of the tree.

        Example usage::

            Tree([[[3],1],[2],0]).unlabelled() # Returns Tree([[[]],[]])
        """
        return Tree(self.unlabelled_repr)

    def color_sequence(self):
        return _list_repr_to_color_sequence(self.list_repr)

######################################
@dataclass(frozen=True)
class CommutativeForest:
    """
    A commutative product of trees.

    :param tree_list: A list of trees contained in the forest

    Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])

            f = CommutativeForest([t1,t2,t3])
    """
######################################
    tree_list : Union[tuple, list] = tuple()
    count : Counter = None
    hash_ : int = None

    def __post_init__(self):
        tuple_repr = tuple(self.tree_list)
        if tuple_repr == tuple():
            tuple_repr = (Tree(None),)
        object.__setattr__(self, 'tree_list', tuple_repr)

    __copy__ = _frozen_copy
    __deepcopy__ = _frozen_deepcopy

    def __hash__(self):
        return _lazy_hash(self, 'tree_list')

    def simplify(self) -> 'Forest':  # Remove redundant empty trees
        """
        Simplify the forest by removing redundant empty trees.

        :return: self
        :rtype: Forest

        Example usage::

            f = Tree([[],[[]]]) * Tree(None)
            f.simplify() #Returns Tree([[],[[]]])
        """
        if len(self.tree_list) <= 1:
            return self

        filtered = tuple(t for t in self.tree_list if t.list_repr is not None)

        if not filtered:
            return EMPTY_FOREST
        if len(filtered) == len(self.tree_list):
            return self
        return Forest(filtered)

    def __repr__(self):
        if len(self.tree_list) == 0:
            return "\u2205"

        r = ""
        for t in self.tree_list[:-1]:
            r += repr(t) + " "
        r += repr(self.tree_list[-1]) + ""
        return r

    def _repr_svg_(self):
        from .display import _to_svg
        return _to_svg(self)

    def __iter__(self):
        yield from self.tree_list

    def join(self, root_color : int = 0) -> 'Tree':
        """
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns the tree :math:`[t_1, t_2, \\cdots, t_k]`.
        In :cite:`connes1999hopf`, this map is denoted by :math:`B_+`.

        :param root_color: Color to assign to the root (default 0)
        :type root_color: int
        :return: :math:`[t_1, t_2, \\cdots, t_k]`
        :rtype: Tree

        Example usage::

            f = Tree([]) * Tree([[]])
            f.join() #Returns Tree([[],[[]]])
        """
        if not isinstance(root_color, int):
            raise TypeError("root_color must be int, not " + str(type(root_color)))

        out = [t.list_repr for t in self.tree_list] + [root_color]
        out = tuple(filter(lambda x: x is not None, out))
        return Tree(out)

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
        return max((t.colors() for t in self.tree_list), default=0)

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

    def sign(self) -> 'ForestSum':
        """
        Returns the forest signed by the number of nodes, :math:`(-1)^{|f|} f`.

        :return: Signed forest, :math:`(-1)^{|f|} f`
        :rtype: ForestSum

        Example usage::

            f1 = Tree([[]]) * Tree([[],[]])
            f1.sign() #Returns - Tree([[]]) * Tree([[],[]])

            f1 = Tree([]) * Tree([[],[]])
            f1.sign() #Returns Tree([]) * Tree([[],[]])
        """
        return self.as_forest_sum() if self.nodes() % 2 == 0 else -self

    def __mul__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> Union['Forest', 'ForestSum']:
        """
        Multiplies a forest by a:

        - scalar, returning a ForestSum
        - Tree, returning a Forest,
        - Forest, returning a Forest,
        - ForestSum, returning a ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum

        Example usage::

            t = 2 * Tree([[]]) * Forest([Tree([]), Tree([[],[]])])
        """
        if _is_scalar(other):
            out = ForestSum(( (other, self), ))
        elif isinstance(other, Tree):
            out = Forest(self.tree_list + (other,))
        elif isinstance(other, Forest):
            out = Forest(self.tree_list + other.tree_list)
        elif isinstance(other, ForestSum):
            out = ForestSum(tuple( (c, self * f) for c, f in other.term_list ))
        else:
            _check_compatible(self, other)
            raise TypeError("Cannot multiply Forest and " + str(type(other)))

        return out.simplify()

    __rmul__ = __mul__

    def __pow__(self, n : int) -> 'Forest':
        """
        Returns the :math:`n^{th}` power of a forest for a positive integer
        :math:`n`, given by a forest with :math:`n` copies of the original forest.

        :param n: Exponent, a positive integer

        Example usage::

            t = ( Tree([]) * Tree([[]]) ) ** 3
        """
        if not isinstance(n, int):
            raise TypeError("Exponent in Forest.__pow__ must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise Forest to a negative power")
        if n == 0:
            return EMPTY_FOREST
        out = Forest(self.tree_list * n)
        return out.simplify()

    def __add__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'ForestSum':
        """
        Adds a forest to a scalar, Tree, Forest or ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 + Tree([[]]) + Forest([Tree([]), Tree([[],[]])])
        """
        if _is_scalar(other):
            out = ForestSum((  (1, self), (other, EMPTY_FOREST)  ))
        elif isinstance(other, (Tree, Forest)):
            out = ForestSum(( (1, self), (1, other) ))
        elif isinstance(other, ForestSum):
            _check_compatible(self, other)
            out = ForestSum( ((1, self),) + other.term_list )
        else:
            _check_compatible(self, other)
            raise TypeError("Cannot add Forest and " + str(type(other)))

        return out.simplify()

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    __radd__ = __add__

    def __neg__(self):
        return self * (-1)

    def equals(self, other_forest):
        _lazy_count(self, 'tree_list')
        _lazy_count(other_forest, 'tree_list')
        return self.count == other_forest.count

    def __eq__(self, other : Union['Forest', 'ForestSum']) -> bool:
        """
        Compares the forest with another object and returns true if they
        represent the same forest, regardless of class type (Forest or ForestSum)
        or possible reorderings of trees.

        :param other: Forest or ForestSum
        :rtype: bool

        Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])
            t4 = Tree([[],[[]]])

            t1 * t2 == t2 * t1 #True
            t1 * t2 == (t1 * t2).as_forest_sum() #True
            t1 * t3 == t1 * t4 #True
        """
        if _is_scalar(other):
            return self.as_forest_sum() == other * EMPTY_TREE
        if isinstance(other, Tree):
            return self.equals(other.as_forest())
        if isinstance(other, CommutativeForest):
            return self.equals(other)
        if isinstance(other, ForestSum):
            return self.as_forest_sum() == other
        return NotImplemented

    def as_forest(self):
        return self

    def as_forest_sum(self) -> 'ForestSum':
        """
        Returns the forest f as a forest sum. Equivalent to ``ForestSum([f])``.

        :return: Forest as a forest sum
        :rtype: ForestSum

        Example usage::

            f = Tree([[],[[]]]) * Tree([[]])
            f.as_forest_sum() #Returns ForestSum([t])
        """
        return ForestSum(( (1,self), ))

    def singleton_reduced(self) -> 'Forest':
        """
        Removes redundant occurrences of the single-node tree in the forest.
        If the forest contains a tree with more than one node, removes all
        occurences of the single-node tree. Otherwise, returns the single-node tree.

        :return: Singleton-reduced forest

        Example usage::

            f1 = Tree([]) * Tree([[],[]])
            f2 = Tree([]) * Tree([]) * Tree([])

            f1.singleton_reduced() #Returns Tree([[],[]])
            f2.singleton_reduced() #Returns Tree([])
        """
        if self.colors() > 1:
            warnings.warn("Singleton reduced representation will not respect colorings")
        out = self.simplify()
        if len(out.tree_list) > 1:
            new_tree_list = tuple(filter(lambda x: len(x.list_repr) != 1, out.tree_list))
            if len(new_tree_list) == 0:
                new_tree_list = (Tree([]),)
            out = Forest(new_tree_list)
        return out

    def __matmul__(self, other: Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'TensorProductSum':
        """
        Returns the tensor product of a Forest and a scalar, Tree, Forest or ForestSum.

        :param other: Other
        :type other: int | float | Tree | Forest | ForestSum
        :return: Tensor product
        :rtype: TensorProductSum

        Example usage::

            Tree([]) @ (Tree([[]]) + Tree([]) * Tree([[],[]])) # Returns 1 [] ⊗ [[]]+1 [] ⊗ [] [[], []]
        """
        if _is_scalar(other):
            return TensorProductSum(( (other, self, EMPTY_FOREST), ))
        if isinstance(other, (Tree, Forest)):
            return TensorProductSum(( (1, self, other.as_forest()), ))
        if isinstance(other, ForestSum):
            term_list = []
            for c, f in other:
                term_list.append((c, self, f))
            return TensorProductSum(term_list)
        raise TypeError("Cannot take tensor product of Forest and " + str(type(other)))

    def __getitem__(self, i):
        return self.tree_list[i]


Forest = CommutativeForest


######################################
@dataclass(frozen=True)
class ForestSum:
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

    def __post_init__(self):
        new_term_list = []

        for term in self.term_list:
            if isinstance(term[1], ForestLike):
                new_term_list.append(term)
            elif isinstance(term[1], Tree):
                new_term_list.append((term[0], term[1].as_forest()))
            elif isinstance(term[1], TreeLike):
                if hasattr(term[1], 'as_ordered_forest'):
                    new_term_list.append((term[0], term[1].as_ordered_forest()))
                else:
                    new_term_list.append((term[0], CommutativeForest((term[1],))))
            else:
                raise TypeError("Terms must be tuples of (coefficient, ForestLike | TreeLike)")

        new_term_list = tuple(new_term_list)
        object.__setattr__(self, 'term_list', new_term_list)

    __copy__ = _frozen_copy
    __deepcopy__ = _frozen_deepcopy

    def __hash__(self):
        return _lazy_hash(self, 'term_list')

    def __repr__(self):
        if len(self.term_list) == 0:
            return "0"

        r = ""
        for c, f in self.term_list:
            term_str = str(c) + " * " + repr(f)
            if c >= 0 and r:
                r += " + " + term_str
            else:
                r += " " + term_str
        return r

    def _repr_svg_(self):
        from .display import _to_svg
        return _to_svg(self)

    def __iter__(self):
        for c,f in self.term_list:
            yield c,f

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
        return max((f.colors() for _, f in self.term_list), default=0)

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


    def simplify(self) -> 'ForestSum':
        """
        Simplify the forest sum by removing redundant empty trees
        and cancelling terms where applicable.

        :return: Reduced forest sum
        :rtype: ForestSum

        Example usage::

            s = Tree([[],[[]]]) * Tree(None) + Tree([]) + Tree([[]]) - Tree([[]])
            s.simplify() #Returns Tree([[],[[]]]) + Tree([])
        """
        merged = {}
        for c, f in self.term_list:
            f_simplified = f.simplify()
            if f_simplified in merged:
                merged[f_simplified] = merged[f_simplified] + c
            else:
                merged[f_simplified] = c
        result = tuple((c, f) for f, c in merged.items() if c != 0)
        if not result:
            return ZERO_FOREST_SUM
        return ForestSum(result)

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

    def sign(self) -> 'ForestSum':
        """
        Returns the forest sum where every forest is replaced by its
        signed value, :math:`(-1)^{|f|} f`.

        :return: Signed forest sum
        :rtype: ForestSum

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.sign() #Returns Tree([[[]],[]]) * Tree([[]]) - 2 * Tree([])
        """
        return ForestSum(tuple((-c if f.nodes() % 2 else c, f) for c,f in self.term_list))

    def __mul__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'ForestSum':
        """
        Multiplies a ForestSum by a scalar, Tree, Forest or ForestSum, returning a ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 * Tree([[]]) * ForestSum([Tree([]), Tree([[],[]])], [1, -2])
        """
        return self._mul_impl(other, reverse=False)

    def __rmul__(self, other):
        return self._mul_impl(other, reverse=True)

    def _mul_impl(self, other, *, reverse):
        if _is_scalar(other):
            new_term_list = tuple( (c * other, f) for c, f in self.term_list )
        elif isinstance(other, (TreeLike, ForestLike)):
            _check_compatible(self, other)
            if reverse:
                new_term_list = tuple( (c, other * f) for c, f in self.term_list )
            else:
                new_term_list = tuple( (c, f * other) for c, f in self.term_list )
        elif isinstance(other, ForestSum):
            _check_compatible(self, other)
            left, right = (other.term_list, self.term_list) if reverse else (self.term_list, other.term_list)
            new_term_list = tuple( (c1 * c2, f1 * f2) for c1, f1 in left for c2, f2 in right if c1 != 0 and c2 != 0)
        else:
            raise TypeError("Cannot multiply ForestSum and " + str(type(other)))

        out = ForestSum(new_term_list) if new_term_list else ZERO_FOREST_SUM
        return out.simplify()


    def __pow__(self, n : int) -> 'ForestSum':
        """
        Returns the :math:`n^{th}` power of a forest sum for a positive integer :math:`n`.

        :param n: Exponent, a positive integer
        :rtype: ForestSum

        Example usage::

            t = ( Tree([]) * Tree([[]]) + Tree([[],[]]) ) ** 3
        """
        if not isinstance(n, int):
            raise TypeError("Exponent in ForestSum.__pow__ must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise ForestSum to a negative power")
        if n == 0:
            return EMPTY_FOREST_SUM

        temp = self
        for _ in range(n-1):
            temp = temp * self

        return temp.simplify()

    def __add__(self, other : Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'ForestSum':
        """
        Adds a ForestSum to a scalar, Tree, Forest or ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 + Tree([[]]) + ForestSum([Tree([]), Tree([[],[]])], [1, -2])
        """
        if _is_scalar(other):
            new_term_list = self.term_list + ((other, EMPTY_FOREST),)
        elif isinstance(other, (TreeLike, ForestLike)):
            _check_compatible(self, other)
            new_term_list = self.term_list + ((1, other),)
        elif isinstance(other, ForestSum):
            _check_compatible(self, other)
            new_term_list = self.term_list + other.term_list
        else:
            raise TypeError("Cannot add ForestSum and " + str(type(other)))

        out = ForestSum(new_term_list)
        return out.simplify()

    def __sub__(self, other):
        return self + (- other)

    def __rsub__(self, other):
        return (-self) + other

    __radd__ = __add__

    def __neg__(self):
        return ForestSum(tuple((-c, f) for c, f in self.term_list))

    def equals(self, other):
        _lazy_count(self, 'term_list')
        _lazy_count(other, 'term_list')
        return self.count == other.count


    def __eq__(self, other : 'ForestSum') -> bool:
        """
        Compares the forest sum with another forest sum and returns true if
        they represent the same forest sum, regardless of possible reorderings
        of trees.

        :param other: ForestSum
        :rtype: bool

        Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])
            t4 = Tree([[],[[]]])

            t1 * t2 + t3 == t3 + t2 * t1 # True
            t1 * t2 + t3 == t1 * t2 + t4 # True
        """
        if _is_scalar(other):
            return self.equals(other * EMPTY_TREE)
        if isinstance(other, Tree):
            return self.equals(other.as_forest_sum())
        if isinstance(other, Forest):
            return self.equals(other.as_forest_sum())
        if isinstance(other, ForestSum):
            return self.equals(other)
        if isinstance(other, TreeLike) and hasattr(other, 'as_forest_sum'):
            return self.equals(other.as_forest_sum())
        if isinstance(other, ForestLike) and hasattr(other, 'as_forest_sum'):
            return self.equals(other.as_forest_sum())
        return NotImplemented

    def singleton_reduced(self) -> 'ForestSum':
        """
        Removes redundant occurrences of the single-node tree in each forest of the
        forest sum. If the forest contains a tree with more than one node, removes
        all occurences of the single-node tree. Otherwise, replaces it with the
        single-node tree.

        :return: Singleton-reduced forest sum
        :rtype: ForestSum

        Example usage::

            s1 = Tree([]) * Tree([[],[]]) + Tree([]) * Tree([]) * Tree([])
            s1.singleton_reduced() #Returns Tree([[],[]]) + Tree([])
        """
        return ForestSum(tuple((c, f.singleton_reduced()) for c, f in self.term_list))

    def as_forest_sum(self):
        return self

    def __matmul__(self, other: Union[int, float, 'Tree', 'Forest', 'ForestSum']) -> 'TensorProductSum':
        """
        Returns the tensor product of a ForestSum and a scalar, Tree, Forest or ForestSum.

        :param other: Other
        :type other: int | float | Tree | Forest | ForestSum
        :return: Tensor product
        :rtype: TensorProductSum

        Example usage::

            Tree([]) @ (Tree([[]]) + Tree([]) * Tree([[],[]])) # Returns 1 [] ⊗ [[]]+1 [] ⊗ [] [[], []]
        """
        if _is_scalar(other):
            term_list = []
            for c, f in self:
                term_list.append((other * c, f, EMPTY_FOREST))
            return TensorProductSum(term_list)
        if isinstance(other, (Tree, Forest)):
            other_ = other.as_forest()
            term_list = []
            for c, f in self:
                term_list.append((c, f, other_))
            return TensorProductSum(term_list)
        if isinstance(other, ForestSum):
            term_list = []
            for c1, f1 in self:
                for c2, f2 in other:
                    term_list.append((c1 * c2, f1, f2))
            return TensorProductSum(term_list)
        raise TypeError("Cannot take tensor product of ForestSum and " + str(type(other)))

    def __getitem__(self, i):
        return self.term_list[i]

##############################################
##############################################

def _is_scalar(obj):
    return isinstance(obj, numbers.Real)

def _is_tree_or_forest(obj):
    return isinstance(obj, (TreeLike, ForestLike))


def _coerce_to_forest(obj):
    """Coerce a tree-like object to its corresponding forest type."""
    if isinstance(obj, ForestLike):
        return obj
    if isinstance(obj, Tree):
        return obj.as_forest()
    if isinstance(obj, PlanarTree):
        return obj.as_ordered_forest()
    raise TypeError(f"Cannot coerce {type(obj)} to forest")

EMPTY_TREE = Tree(None)
EMPTY_FOREST = Forest((EMPTY_TREE,))
EMPTY_FOREST_SUM = ForestSum( ( (1, EMPTY_FOREST), ) )
ZERO_FOREST_SUM = ForestSum(())

##############################################
##############################################

@dataclass(frozen=True)
class TensorProductSum:
    """
    A linear combination of tensor products of forests.

    :param term_list: A list of tuples representing terms in the sum.
        Tuples must be of the form `(c, f1, f2)`, where `c` is an `int`
        or `float` and `f1, f2` are Forests, representing the term
        :math:`c \\cdot (f1 \\otimes f2)`.

    Example usage::

            tp = Tree([]) @ Tree([[]]) - 2 * Tree([[],[]]) @ Tree(None)
    """
    term_list: Union[tuple, list, None] #(c, f1, f2)
    count : Counter = None
    hash_ : int = None

    def __post_init__(self):
        tuple_list = []
        for x in self.term_list:
            if not (_is_scalar(x[0]) and _is_tree_or_forest(x[1]) and _is_tree_or_forest(x[2])):
                raise TypeError("Terms must be tuples of type (scalar, TreeLike | ForestLike, TreeLike | ForestLike)")
            tuple_list.append((x[0], _coerce_to_forest(x[1]), _coerce_to_forest(x[2])))
        tuple_list = tuple(tuple_list)
        object.__setattr__(self, 'term_list', tuple_list)

    __copy__ = _frozen_copy
    __deepcopy__ = _frozen_deepcopy

    def __repr__(self):
        if self.term_list is None or self.term_list == tuple():
            return "0"

        r = ""
        for c, f1, f2 in self.term_list:
            term_str = str(c) + " * " + repr(f1) + " \u2297 " + repr(f2)
            if c >= 0 and r:
                r += " + " + term_str
            else:
                r += " " + term_str
        return r

    def _repr_svg_(self):
        from .display import _to_svg
        return _to_svg(self)

    def simplify(self) -> 'TensorProductSum':
        """
        Simplify the tensor product sum by removing redundant empty trees
        and cancelling terms where applicable.

        :return: Reduces tensor product sum
        :rtype: TensorProductSum

        Example usage::

            tp = Tree([[],[[]]]) @ (Tree([]) * Tree(None)) + Tree([]) @ Tree([[]]) - Tree([]) @ Tree([[]])
            tp.simplify() #Returns 1 [[], [[]]] ⊗ []
        """
        new_term_list = []

        for c, f1, f2 in self.term_list:
            f1_reduced = f1.simplify()
            f2_reduced = f2.simplify()

            for i, (_, f1_, f2_) in enumerate(new_term_list):
                if f1_reduced.equals(f1_) and f2_reduced.equals(f2_):
                    old_term_ = new_term_list[i]
                    new_term_list[i] = (old_term_[0] + c, old_term_[1], old_term_[2])
                    break
            else:
                new_term_list.append((c, f1_reduced, f2_reduced))

        result = tuple(term for term in new_term_list if term[0] != 0)
        return TensorProductSum(result)

    def singleton_reduced(self) -> 'TensorProductSum':
        """
        Removes redundant occurrences of the single-node tree in each forest of the
        tensor product sum. If the forest contains a tree with more than one node, removes
        all occurences of the single-node tree. Otherwise, replaces it with the
        single-node tree.

        :return: Singleton-reduced tensor product sum
        :rtype: TensorProductSum

        Example usage::

            s1 = (Tree([]) * Tree([[],[]])) @ (Tree([]) * Tree([]) * Tree([]))
            s1.singleton_reduced() #Returns Tree([[],[]]) @ Tree([])
        """
        return TensorProductSum(tuple((c, f1.singleton_reduced(), f2.singleton_reduced()) for c, f1, f2 in self.term_list))

    def __eq__(self, other : 'TensorProductSum') -> bool:
        """
        Compares the tensor product sum with another tensor product sum and returns true if
        they represent the same sum, regardless of possible reorderings of trees within forests
        or reorderings of terms.

        :param other: TensorProductSum
        :rtype: bool

        Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])
            t4 = Tree([[],[[]]])

            t1 @ t2 + t2 @ t3 == t2 @ t3 + t1 @ t2 # True
            t1 @ (t2 * t3) == t1 @ (t3 * t2) # True
            t1 @ t3 == t1 @ t4 # True
        """
        if not isinstance(other, TensorProductSum):
            return NotImplemented
        _lazy_count(self, 'term_list')
        _lazy_count(other, 'term_list')
        return self.count == other.count

    def __hash__(self):
        return _lazy_hash(self, 'term_list')

    def __add__(self, other : 'TensorProductSum') -> 'TensorProductSum':
        """
        Adds two tensor product sums.

        :param other: Other tensor product sum
        :type other: TensorProductSum
        """
        if not isinstance(other, TensorProductSum):
            raise TypeError("Cannot add TensorSum and " + str(type(other)))
        return TensorProductSum(self.term_list + other.term_list)

    def __neg__(self):
        return TensorProductSum(tuple((-x[0], x[1], x[2]) for x in self.term_list))

    def __sub__(self, other):
        if not isinstance(other, TensorProductSum):
            raise TypeError("Cannot subtract " + str(type(other)) + " from TensorSum")
        return self + (-other)

    def __mul__(self, other : Union[int, float, 'TensorProductSum']) -> 'TensorProductSum':
        """
        Multiplies a tensor product sum by a scalar or tensor product sum.

        :param other: Other
        :type other: int | float | TensorProductSum
        """
        if isinstance(other, TensorProductSum):
            new_term_list = []
            for c1, f11, f12 in self:
                for c2, f21, f22 in other:
                    new_term_list.append((c1 * c2, f11 * f21, f12 * f22))
            return TensorProductSum(tuple(new_term_list))
        if _is_scalar(other):
            return TensorProductSum(tuple((other * x[0], x[1], x[2]) for x in self.term_list))
        raise TypeError("Cannot multiply TensorSum by " + str(type(other)))

    def __rsub__(self, other):
        return (-self) + other

    __radd__ = __add__
    __rmul__ = __mul__

    def __iter__(self):
        for c, f1, f2 in self.term_list:
            yield c, f1, f2

    def __len__(self):
        return len(self.term_list)

    def __getitem__(self, i):
        return self.term_list[i]

    def colors(self):
        """
        Returns the number of colors/labels in the tensor product sum. Since the labels are
        indexed starting from 0, this is equivalent to one more than the maximum label.

        :return: Number of colors
        :rtype: int

        Example usage::

            (Tree([[9],0]) @ Tree([3]) + Tree([2]) @ Tree([4])).colors() # Returns 10
        """
        return max((max(f1.colors(), f2.colors()) for _, f1, f2 in self.term_list), default=0)


######################################
@dataclass(frozen=True)
class PlanarTree:
    """Ordered rooted tree; sibling order is part of identity."""

    list_repr: Union[tuple, list, None] = None
    unlabelled_repr = None

    def __post_init__(self) -> None:
        if self.list_repr is not None:
            if not _check_valid(self.list_repr):
                raise ValueError(f"{self.list_repr!r} is not a valid planar tree representation.")
            tuple_repr: tuple = _to_labelled_tuple(self.list_repr)
            object.__setattr__(self, "list_repr", tuple_repr)
            object.__setattr__(self, "unlabelled_repr", _to_unlabelled_tuple(tuple_repr))
            object.__setattr__(self, "_max_color", _get_max_color(tuple_repr))

    def nodes(self) -> int:
        return _nodes(self.unlabelled_repr)

    def factorial(self) -> int:
        """Compute the tree factorial for a planar tree. Uses the same recursion as ``Tree.factorial()``."""
        return _factorial(self.unlabelled_repr)[0]

    def sigma(self) -> int:
        """Symmetry factor of an ordered tree — always 1 (sibling order is part of identity)."""
        return 1

    def height(self) -> int:
        """Returns the height of the tree (longest root-to-leaf path length)."""
        return _height(self.unlabelled_repr)

    def density(self) -> float:
        """Density of the tree, ``factorial() / nodes()!``."""
        return self.factorial() / math.factorial(self.nodes())

    def alpha(self) -> int:
        """Number of monotone labellings (up to symmetry). Since sigma=1 for planar trees, equals ``beta() / factorial()``."""
        return self.beta() // self.factorial()

    def beta(self) -> int:
        """Number of distinct labellings (up to symmetry). Since sigma=1 for planar trees, equals ``nodes()!``."""
        return math.factorial(self.nodes())

    def unjoin(self) -> 'NoncommutativeForest':
        """For a tree t = [t_1, ..., t_k], returns the forest t_1 ... t_k (the B- map)."""
        if self.list_repr is None:
            return EMPTY_ORDERED_FOREST
        return NoncommutativeForest(tuple(PlanarTree(rep) for rep in self.list_repr[:-1]))

    def unlabelled(self) -> 'PlanarTree':
        """Returns the unlabelled version of the tree."""
        return PlanarTree(self.unlabelled_repr)

    def as_ordered_forest(self) -> 'OrderedForest':
        return OrderedForest((self,))

    def to_nonplanar_tree(self) -> Tree:
        if self.list_repr is None:
            return Tree(None)
        return Tree(self.list_repr)

    def __hash__(self):
        return hash(self.list_repr)

    def __eq__(self, other):
        if isinstance(other, PlanarTree):
            return self.list_repr == other.list_repr
        if isinstance(other, (NoncommutativeForest, ForestSum)):
            return self.as_forest_sum() == other
        return NotImplemented

    def __mul__(self, other):
        return self.as_ordered_forest().__mul__(other)

    def __rmul__(self, other):
        return self.as_ordered_forest().__rmul__(other)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Exponent must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise PlanarTree to a negative power")
        if n == 0:
            return EMPTY_ORDERED_FOREST
        return NoncommutativeForest((self,) * n).simplify()

    def __add__(self, other):
        if _is_scalar(other):
            return ForestSum(((1, self.as_ordered_forest()), (other, EMPTY_ORDERED_FOREST)))
        if isinstance(other, (PlanarTree, NoncommutativeForest)):
            return ForestSum(((1, self), (1, other)))
        if isinstance(other, ForestSum):
            _check_compatible(self, other)
            return ForestSum(((1, self),) + other.term_list)
        _check_compatible(self, other)
        raise TypeError("Cannot add PlanarTree and " + str(type(other)))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return ForestSum(((-1, self.as_ordered_forest()),))

    def sign(self):
        return self.as_forest_sum() if self.nodes() % 2 == 0 else -self

    def colors(self) -> int:
        """Returns the number of colors/labels in a labelled planar tree."""
        if self.list_repr is None:
            return 0
        return self._max_color + 1

    def __repr__(self):
        if self.list_repr is None:
            return "\u2205"
        if self._max_color == 0:
            return repr(_to_list(self.unlabelled_repr))
        return repr(_to_list(self.list_repr))

    def _repr_svg_(self):
        from .display import _to_svg
        return _to_svg(self)

    def level_sequence(self) -> list:
        return _list_repr_to_level_sequence(self.unlabelled_repr)

    def color_sequence(self):
        return _list_repr_to_color_sequence(self.list_repr)

    def __matmul__(self, other):
        if _is_scalar(other):
            return TensorProductSum(((other, self.as_ordered_forest(), EMPTY_ORDERED_FOREST),))
        if isinstance(other, (PlanarTree, NoncommutativeForest)):
            return TensorProductSum(((1, self.as_ordered_forest(), _coerce_to_forest(other)),))
        if isinstance(other, ForestSum):
            return TensorProductSum(tuple((c, self, f) for c, f in other))
        raise TypeError("Cannot take tensor product of PlanarTree and " + str(type(other)))

    def sorted_list_repr(self):
        """Returns the list representation. For planar trees this is the identity
        (sibling order is part of the tree's identity)."""
        return self.list_repr

    def equals(self, other_tree):
        """Two planar trees are equal iff their list representations match exactly."""
        return self.list_repr == other_tree.list_repr

    def __next__(self) -> 'PlanarTree':
        """Generates the next planar tree in lexicographic order of level sequences.

        :return: Next planar tree
        :rtype: PlanarTree

        Example usage::

                t = PlanarTree([[],[]])
                next(t) # returns PlanarTree([[[]]])
        """
        if self.list_repr is None:
            return PlanarTree([])
        if self._max_color > 0:
            warnings.warn("Calling next() on a labelled tree will ignore the labelling.")

        layout = self.level_sequence()
        next_ = _next_planar_layout(layout)
        return PlanarTree(_level_sequence_to_list_repr(next_))

    def as_forest_sum(self):
        return ForestSum(((1, self.as_ordered_forest()),))


@dataclass(frozen=True)
class NoncommutativeForest:
    """Noncommutative forest (word) of planar trees."""

    tree_list: tuple[PlanarTree, ...] = tuple()

    def __post_init__(self) -> None:
        values: tuple[PlanarTree, ...] = tuple(self.tree_list)
        if len(values) == 0:
            values = (EMPTY_PLANAR_TREE,)
        object.__setattr__(self, "tree_list", values)

    def __iter__(self) -> Iterator[PlanarTree]:
        yield from self.tree_list

    def __getitem__(self, index: int) -> PlanarTree:
        return self.tree_list[index]

    def simplify(self) -> 'NoncommutativeForest':
        if len(self.tree_list) <= 1:
            return self
        filtered = tuple(tree for tree in self.tree_list if tree.list_repr is not None)
        if len(filtered) == 0:
            return EMPTY_ORDERED_FOREST
        if len(filtered) == len(self.tree_list):
            return self
        return NoncommutativeForest(filtered)

    def nodes(self) -> int:
        return sum(tree.nodes() for tree in self.tree_list)

    def num_trees(self) -> int:
        """Returns the number of trees in the forest."""
        return len(self.tree_list)

    def singleton_reduced(self) -> 'NoncommutativeForest':
        """Remove single-node trees from the forest, preserving order."""
        if self.colors() > 1:
            warnings.warn("Singleton reduced representation will not respect colorings")
        out = self.simplify()
        if len(out.tree_list) > 1:
            new_tree_list = tuple(t for t in out.tree_list if len(t.list_repr) != 1)
            if len(new_tree_list) == 0:
                new_tree_list = (PlanarTree([]),)
            out = NoncommutativeForest(new_tree_list)
        return out

    def as_forest(self):
        """Return self (protocol compatibility)."""
        return self

    def factorial(self) -> int:
        """Product of tree factorials: ``prod(t.factorial() for t in self.tree_list)``."""
        return math.prod(t.factorial() for t in self.tree_list)

    def __repr__(self):
        if len(self.tree_list) == 0:
            return "\u2205"
        return " ".join(repr(t) for t in self.tree_list)

    def _repr_svg_(self):
        from .display import _to_svg
        return _to_svg(self)

    def colors(self) -> int:
        return max((t.colors() for t in self.tree_list), default=0)

    def equals(self, other):
        if not isinstance(other, NoncommutativeForest):
            return False
        return self.tree_list == other.tree_list

    def _forest_mul(self, other, *, prepend):
        if _is_scalar(other):
            return ForestSum(((sympy.sympify(other), self),))
        if isinstance(other, PlanarTree):
            other_trees = (other,)
        elif isinstance(other, NoncommutativeForest):
            other_trees = other.tree_list
        elif isinstance(other, ForestSum):
            _check_compatible(self, other)
            terms = tuple(
                (coeff, NoncommutativeForest(
                    (forest.tree_list + self.tree_list) if prepend
                    else (self.tree_list + forest.tree_list)
                ).simplify())
                for coeff, forest in other.term_list
            )
            return ForestSum(terms).simplify()
        else:
            _check_compatible(self, other)
            side = (f"{type(other)} and NoncommutativeForest" if prepend
                    else f"NoncommutativeForest and {type(other)}")
            raise TypeError(f"Cannot multiply {side}")
        trees = (other_trees + self.tree_list) if prepend else (self.tree_list + other_trees)
        return NoncommutativeForest(trees).simplify()

    def __mul__(self, other):
        return self._forest_mul(other, prepend=False)

    def __rmul__(self, other):
        return self._forest_mul(other, prepend=True)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Exponent must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise NoncommutativeForest to a negative power")
        if n == 0:
            return EMPTY_ORDERED_FOREST
        return NoncommutativeForest(self.tree_list * n).simplify()

    def __add__(self, other):
        if _is_scalar(other):
            return ForestSum(((1, self), (other, EMPTY_ORDERED_FOREST)))
        if isinstance(other, (PlanarTree, NoncommutativeForest)):
            return ForestSum(((1, self), (1, other)))
        if isinstance(other, ForestSum):
            _check_compatible(self, other)
            return ForestSum(((1, self),) + other.term_list)
        _check_compatible(self, other)
        raise TypeError("Cannot add NoncommutativeForest and " + str(type(other)))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return ForestSum(((-1, self),))

    def __eq__(self, other):
        if isinstance(other, NoncommutativeForest):
            return self.tree_list == other.tree_list
        if isinstance(other, (PlanarTree, ForestSum)):
            return self.as_forest_sum() == other
        return NotImplemented

    def __hash__(self):
        return hash(self.tree_list)

    def sign(self):
        return self.as_forest_sum() if self.nodes() % 2 == 0 else -self

    def __matmul__(self, other):
        if _is_scalar(other):
            return TensorProductSum(((other, self, EMPTY_ORDERED_FOREST),))
        if isinstance(other, (PlanarTree, NoncommutativeForest)):
            return TensorProductSum(((1, self, _coerce_to_forest(other)),))
        if isinstance(other, ForestSum):
            return TensorProductSum(tuple((c, self, f) for c, f in other))
        raise TypeError("Cannot take tensor product of NoncommutativeForest and " + str(type(other)))

    def as_forest_sum(self):
        return ForestSum(((1, self),))

    def join(self, root_color=0):
        children = tuple(t.list_repr for t in self.tree_list if t.list_repr is not None)
        return PlanarTree(children + (root_color,))


OrderedForest = NoncommutativeForest
OrderedForestSum = ForestSum


EMPTY_PLANAR_TREE = PlanarTree(None)
EMPTY_ORDERED_FOREST = NoncommutativeForest((EMPTY_PLANAR_TREE,))

_CROSS_TYPE_HINT = (
    "Cannot combine planar and non-planar tree types. "
    "Use PlanarTree/OrderedForest with planar algebras (pgl, pbck), "
    "or Tree/Forest with non-planar algebras (gl, bck, cem)."
)

def _is_planar_obj(obj):
    """Return True if obj is planar, False if non-planar, None if unknown."""
    if isinstance(obj, (PlanarTree, NoncommutativeForest)):
        return True
    if isinstance(obj, (Tree, CommutativeForest)):
        return False
    if isinstance(obj, ForestSum):
        for c, f in obj.term_list:
            if isinstance(f, NoncommutativeForest):
                return True
            if isinstance(f, CommutativeForest):
                return False
        return None
    return None

def _check_compatible(a, b):
    """Raise TypeError if a and b mix planar and non-planar types."""
    pa, pb = _is_planar_obj(a), _is_planar_obj(b)
    if pa is not None and pb is not None and pa != pb:
        raise TypeError(_CROSS_TYPE_HINT)


def validate_order(order: int, *, allow_zero: bool = True) -> None:
    if allow_zero:
        if order < 0:
            raise ValueError("order must be non-negative")
        return
    if order <= 0:
        raise ValueError("order must be positive")
