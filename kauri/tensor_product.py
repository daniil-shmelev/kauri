from dataclasses import dataclass
from collections import Counter
from typing import Union
from .abstract_tree import _is_scalar, _is_tree_or_forest

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
                raise TypeError("Terms must be tuples of type (int | float, Tree | Forest, Tree | Forest)")
            tuple_list.append((x[0], x[1].as_forest(), x[2].as_forest()))
        tuple_list = tuple(tuple_list)
        object.__setattr__(self, 'term_list', tuple_list)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        memodict[id(self)] = self
        return self

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

    def _set_counter(self):
        if self.count is None:
            object.__setattr__(self, 'count', Counter(self.simplify().term_list))

    def _set_hash(self):
        self._set_counter()
        if self.hash_ is None:
            object.__setattr__(self, 'hash_', hash(frozenset(self.count.items())))

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
            raise TypeError("Cannot check equality of TensorSum and " + str(type(other)))
        self._set_counter()
        other._set_counter()
        return self.count == other.count

    def __hash__(self):
        self._set_hash()
        return self.hash_

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
        if isinstance(other, (int, float)):
            return TensorProductSum(tuple((other * x[0], x[1], x[2]) for x in self.term_list))
        raise TypeError("Cannot multiply TensorSum by " + str(type(other)))

    __radd__ = __add__
    __rsub__ = __sub__
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
        return max(max(f1.colors(), f2.colors()) for _, f1, f2 in self.term_list)