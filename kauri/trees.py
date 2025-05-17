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

from .abstract_tree import AbstractTree, AbstractForest, AbstractForestSum
from .tensor_product import TensorProductSum

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

    def __post_init__(self):
        if self.list_repr is not None:
            if not _check_valid(self.list_repr):
                raise ValueError(repr(self.list_repr) + " is not a valid list representation for a tree.")
            tuple_repr = _to_labelled_tuple(self.list_repr)
            object.__setattr__(self, 'list_repr', tuple_repr)
            unlabelled_repr = _to_unlabelled_tuple(tuple_repr)
            object.__setattr__(self, 'unlabelled_repr', unlabelled_repr)
            object.__setattr__(self, '_max_color', _get_max_color(tuple_repr))

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        memodict[id(self)] = self
        return self

    def __repr__(self):
        if self.list_repr is None:
            return "\u2205"
        if self._max_color == 0:
            return repr(_to_list(self.unlabelled_repr))
        return repr(_to_list(self.list_repr))

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
        if isinstance(other, (int, float)):
            out = ForestSum(( (other,self),  ))
        elif isinstance(other, Tree):
            out = Forest((self, other))
        elif isinstance(other, Forest):
            out = Forest((self,) + other.tree_list)
        elif isinstance(other, ForestSum):
            out = ForestSum(tuple((c, self * f) for c,f in other.term_list))
        else:
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
        if isinstance(other, (int, float)):
            out = ForestSum((  (1, self), (other, EMPTY_FOREST)  ))
        elif isinstance(other, (Tree, Forest)):
            out = ForestSum((  (1, self), (1, other)  ))
        elif isinstance(other, ForestSum):
            out = ForestSum( ((1, self),) + other.term_list )
        else:
            raise TypeError("Cannot add Tree and " + str(type(other)))

        return out.simplify()

    def __sub__(self, other):
        return self + (-other)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        temp = self * (-1)
        return temp

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
        if isinstance(other, (int, float)):
            return self.as_forest_sum() == other * EMPTY_TREE
        if isinstance(other, Tree):
            return self.equals(other)
        if isinstance(other, Forest):
            return self.as_forest() == other
        if isinstance(other, ForestSum):
            return self.as_forest_sum() == other
        raise TypeError("Cannot check equality of Tree and " + str(type(other)))

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
        if isinstance(other, (int, float)):
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

    def __post_init__(self):
        tuple_repr = tuple(self.tree_list)
        if tuple_repr == tuple():
            tuple_repr = (Tree(None),)
        object.__setattr__(self, 'tree_list', tuple_repr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        memodict[id(self)] = self
        return self

    def _set_counter(self):
        if self.count is None:
            object.__setattr__(self, 'count', Counter(self.simplify().tree_list))

    def _set_hash(self):
        self._set_counter()
        if self.hash_ is None:
            object.__setattr__(self, 'hash_', hash(frozenset(self.count.items())))

    def __hash__(self):
        self._set_hash()
        return self.hash_

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
        if isinstance(other, (int, float)):
            out = ForestSum(( (other, self), ))
        elif isinstance(other, Tree):
            out = Forest(self.tree_list + (other,))
        elif isinstance(other, Forest):
            out = Forest(self.tree_list + other.tree_list)
        elif isinstance(other, ForestSum):
            out = ForestSum(tuple( (c, self * f) for c, f in other.term_list ))
        else:
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
        if isinstance(other, (int, float)):
            out = ForestSum((  (1, self), (other, EMPTY_FOREST)  ))
        elif isinstance(other, (Tree, Forest)):
            out = ForestSum(( (1, self), (1, other) ))
        elif isinstance(other, ForestSum):
            out = ForestSum( ((1, self),) + other.term_list )
        else:
            raise TypeError("Cannot add Forest and " + str(type(other)))

        return out.simplify()

    def __sub__(self, other):
        return self + (-other)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self * (-1)

    def equals(self, other_forest):
        self._set_counter()
        other_forest._set_counter()
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
        if isinstance(other, (int, float)):
            return self.as_forest_sum() == other * EMPTY_TREE
        if isinstance(other, Tree):
            return self.equals(other.as_forest())
        if isinstance(other, Forest):
            return self.equals(other)
        if isinstance(other, ForestSum):
            return self.as_forest_sum() == other
        raise TypeError("Cannot check equality of Forest and " + str(type(other)))

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
        if isinstance(other, (int, float)):
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

    def __post_init__(self):
        new_term_list = []

        for term in self.term_list:
            if isinstance(term[1], Forest):
                new_term_list.append(term)
            elif isinstance(term[1], Tree):
                new_term_list.append((term[0], term[1].as_forest()))
            else:
                raise TypeError("Terms must be tuples of type (int | float, Tree | Forest)")

            if not isinstance(term[0], (int, float)):
                raise TypeError("Terms must be tuples of type (int | float, Tree | Forest)")

        new_term_list = tuple(new_term_list)
        object.__setattr__(self, 'term_list', new_term_list)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        memodict[id(self)] = self
        return self

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

    def __iter__(self):
        for c,f in self.term_list:
            yield c,f

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
        new_forest_list = []
        new_coeff_list = []

        for c, f in self.term_list:
            f_reduced = f.simplify()

            for i, f2 in enumerate(new_forest_list):
                if f_reduced.equals(f2):
                    new_coeff_list[i] += c
                    break
            else:
                new_forest_list.append(f_reduced)
                new_coeff_list.append(c)

        result = tuple((c, f) for c, f in zip(new_coeff_list, new_forest_list) if c != 0)

        if not result:
            return ZERO_FOREST_SUM
        return ForestSum(result)

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
        if isinstance(other, (int, float)):
            new_term_list = tuple( (c * other, f) for c, f in self.term_list )
        elif isinstance(other, (Tree, Forest)):
            new_term_list = tuple( (c, f * other) for c, f in self.term_list )
        elif isinstance(other, ForestSum):
            new_term_list = tuple( (c1 * c2, f1 * f2) for c1, f1 in self.term_list for c2, f2 in other.term_list)
        else:
            raise TypeError("Cannot multiply ForestSum and " + str(type(object)))

        out = ForestSum(new_term_list)
        return out.simplify()

    __rmul__ = __mul__


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
        if isinstance(other, (int, float)):
            new_term_list = self.term_list + ((other, EMPTY_FOREST),)
        elif isinstance(other, (Tree, Forest)):
            new_term_list = self.term_list + ((1, other),)
        elif isinstance(other, ForestSum):
            new_term_list = self.term_list + other.term_list
        else:
            raise TypeError("Cannot add ForestSum and " + str(type(other)))

        out = ForestSum(new_term_list)
        return out.simplify()

    def __sub__(self, other):
        return self + (- other)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self * (-1)

    def equals(self, other):
        self._set_counter()
        other._set_counter()
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
        if isinstance(other, (int, float)):
            return self.equals(other * EMPTY_TREE)
        if isinstance(other, Tree):
            return self.equals(other.as_forest_sum())
        if isinstance(other, Forest):
            return self.equals(other.as_forest_sum())
        if isinstance(other, ForestSum):
            return self.equals(other)
        raise TypeError("Cannot check equality of ForestSum and " + str(type(other)))

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
        if isinstance(other, (int, float)):
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

EMPTY_TREE = Tree(None)
EMPTY_FOREST = Forest((EMPTY_TREE,))
EMPTY_FOREST_SUM = ForestSum( ( (1, EMPTY_FOREST), ) )
ZERO_FOREST_SUM = ForestSum( ( (0, EMPTY_FOREST), ) )
