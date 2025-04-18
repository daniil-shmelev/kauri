import itertools
import copy
import math
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Union, Optional
from functools import cache
from .utils import _nodes, _height, _factorial, _sigma, _sorted_list_repr, _list_repr_to_level_sequence, _to_tuple, _to_list, _next_layout, _level_sequence_to_list_repr

def _counit(t):
    if t.list_repr is None:
        return EMPTY_FOREST_SUM
    else:
        return ZERO_FOREST_SUM

######################################
@dataclass(frozen=True)
class Tree():
    """
    A single planar rooted tree.

    :param list_repr: The nested list representation of the tree

    Example usage::

            t = Tree([[[]],[]])
    """
######################################
    list_repr: Union[tuple, list, None]

    def __post_init__(self):
        tuple_repr = _to_tuple(self.list_repr)
        object.__setattr__(self, 'list_repr', tuple_repr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        memodict[id(self)] = self
        return self

    def __repr__(self):
        return repr(_to_list(self.list_repr)) if self.list_repr is not None else "\u2205"

    def __hash__(self):
        return hash(self.sorted_list_repr())

    def unjoin(self):
        """
        For a tree :math:`t = [t_1, t_2, ..., t_k]`, returns the forest :math:`t_1 t_2 \\cdots t_k`

        :return: :math:`t_1 t_2 \\cdots t_k`
        :rtype: Forest

        Example usage::

            t = Tree([[[]],[]])
            t.unjoin() #Returns Tree([[]]) * Tree([])
        """
        if self.list_repr is None:
            return EMPTY_FOREST
        return Forest(tuple(Tree(rep) for rep in self.list_repr))

    def nodes(self):
        """
        Returns the number of nodes in a tree, :math:`|t|`

        :return: Number of nodes, :math:`|t|`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.nodes() #Returns 4
        """
        return _nodes(self.list_repr)

    def height(self):
        """
        Returns the height of a tree, given by the number of nodes in the longest walk from the root to a leaf.

        :return: Height
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.height() #Returns 3
        """
        return _height(self.list_repr)

    def factorial(self):
        """
        Compute the tree factorial, :math:`t!`

        :return: Tree factorial, :math:`t!`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.factorial() #Returns 8
        """
        return _factorial(self.list_repr)[0]

    def sigma(self):
        """
        Computes the symmetry factor :math:`\\sigma(t)`, the order of the symmetric group of the tree. For a tree
        :math:`t = [t_1^{m_1} t_2^{m_2} \\cdots t_k^{m_k}]`, the symmetry factor satisfies the recursion

        .. math::
            \\sigma(t) = \\prod_{i=1}^k m_i! \\sigma(t_i)^{m_i}.

        :return: Symmetry factor, :math:`\\sigma(t)`
        :rtype: int

        Example usage::

            t = Tree([[[]],[]])
            t.sigma()
        """
        return _sigma(self.list_repr)

    def alpha(self):
        """
        For a tree :math:`t` with :math:`n` nodes, computes the number of distinct ways of labelling the nodes of the tree
        with symbols :math:`\\{1, 2, \\ldots, n\\}`, such that:

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
        return self.beta() / self.factorial()

    def beta(self):
        """
        For a tree :math:`t` with :math:`n` nodes, computes the number of distinct ways of labelling the nodes of the tree
        with symbols :math:`\\{1, 2, \\ldots, n\\}`, such that:

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
        return math.factorial(self.nodes()) / self.sigma()

    def coproduct(self):
        """
        Returns the coproduct of a tree,

        .. math::

            \\Delta(t) = \\sum P_c(t) \\otimes R_c(t)

        :return: Values of :math:`R_c(t)`
        :rtype: list
        :return: Values of :math:`P_c(t)`
        :rtype: list

        Example usage::

            t = Tree([[[]],[]])
            t.split()
        """
        if self.list_repr is None:
            return [EMPTY_TREE], [EMPTY_FOREST]
        if self.list_repr == tuple():
            return [SINGLETON_TREE, EMPTY_TREE], [EMPTY_FOREST, SINGLETON_FOREST]

        tree_list = []
        forest_list = []
        for rep in self.list_repr:
            t = Tree(rep)
            s, b = t.coproduct()
            tree_list.append(s)
            forest_list.append(b)

        new_tree_list = [EMPTY_TREE]
        new_forest_list = [Forest((self,))]

        for p in itertools.product(*tree_list):
            new_tree_list.append(Tree(tuple(t.list_repr for t in p if t.list_repr is not None)))

        for p in itertools.product(*forest_list):
            t = []
            for f in p:
                t += f.tree_list
            new_forest_list.append(Forest(t))

        return new_tree_list, new_forest_list

    def cem_coproduct(self):
        """
        Computes the Calaque, Ebrahimi-Fard and Manchon :cite:`calaque2011two` coproduct on trees, given by

        .. math::

            \\Delta_{CEM}(t) = \\sum (t \\setminus p) \\otimes p_t,

        where, for a tree :math:`t` with edge set :math:`E`, the sum is over all sets of edges :math:`p \\subset E`, and

        - :math:`t \\setminus p` is the forest formed by removing all edges in :math:`p` from the tree :math:`t`,
        - :math:`p_t` is the tree formed by contracting each tree of :math:`t \\setminus p` to a single vertex and re-establishing the edges in :math:`p`. :cite:`chartier2010algebraic`

        :return: Values of :math:`p_t`
        :rtype: list
        :return: Values of :math:`t \\setminus p`
        :rtype: list

        Example usage::

            t = Tree([[[]],[]])
            t.partition()
        """
        if self.list_repr is None:
            raise
        if self.list_repr == tuple():
            return [SINGLETON_TREE], [SINGLETON_FOREST]

        tree_list = []
        forest_list = []
        for rep in self.list_repr:
            t = Tree(rep)
            s, b = t.cem_coproduct()
            tree_list.append(s)
            forest_list.append(b)

        new_tree_list = []
        new_forest_list = []

        num_branches = len(tree_list)

        for edges in itertools.product([0, 1], repeat=num_branches):

            for p in itertools.product(*tree_list):
                rep = []
                for i,t in enumerate(p):
                    if t.list_repr is None:
                        continue
                    if edges[i]:
                        rep += t.list_repr
                    else:
                        rep += [t.list_repr]
                new_tree_list.append(Tree(rep))

            for p in itertools.product(*forest_list):
                #Must ensure that the first tree in the forest is connected to the root
                #If no such tree, add an empty tree to the forest to signify this
                #Forest constructor does not call Forest.reduce(), meaning this empty tree will survive
                t = []
                root_tree_repr = []
                for i,f in enumerate(p):
                    if edges[i]:
                        root_tree_repr += [f.tree_list[0].list_repr]
                        t += f.tree_list[1:]
                    else:
                        t += f.tree_list
                t = [Tree(root_tree_repr)] + t
                new_forest_list.append(Forest(t))

        return new_tree_list, new_forest_list

    @cache
    def antipode(self):
        """
        Returns the antipode of a tree,

        .. math::

            S(t) = -t-\\sum S(P_c(t)) R_c(t)

        :return: Antipode, :math:`S(t)`
        :rtype: ForestSum

        Example usage::

            t = Tree([[[]],[]])
            t.antipode()
        """
        if self.list_repr is None:
            return EMPTY_FOREST_SUM
        elif self.list_repr == tuple():
            return -SINGLETON_FOREST_SUM
        
        subtrees, branches = self.coproduct()
        out = -self.as_forest_sum()
        for i in range(len(subtrees)):
            if subtrees[i]._equals(self) or subtrees[i]._equals(EMPTY_TREE):
                continue
            out = out - branches[i].antipode() * subtrees[i]

        return out.reduce()

    @cache
    def cem_antipode(self):
        # TODO
        if self.list_repr is None:
            return ZERO_FOREST_SUM
        elif self.list_repr == tuple():
            return SINGLETON_FOREST_SUM

        subtrees, branches = self.cem_coproduct()
        out = -self.as_forest_sum()
        for i in range(len(subtrees)):
            if branches[i]._equals(self.as_forest()) or subtrees[i]._equals(self):
                continue
            out = out - subtrees[i].cem_antipode() * branches[i]

        return out.singleton_reduced().reduce()

    def sign(self):
        """
        Returns the tree signed by the number of nodes, :math:`(-1)^{|t|} t`.

        :return: Signed tree, :math:`(-1)^{|t|} t`
        :rtype: ForestSum

        Example usage::

            t = Tree([[[]],[]])
            t.sign()
        """
        return self if self.nodes() % 2 == 0 else -self

    def signed_antipode(self):
        """
        Returns the antipode of the signed tree, :math:`S((-1)^{|t|} t)`.

        :return: Antipode of the signed tree, :math:`S((-1)^{|t|} t)`
        :rtype: ForestSum

        .. note::
            Since the antipode and sign functions commute, this function is equivalent to both ``self.sign().antipode()`` and
            ``self.antipode().sign()``.

        Example usage::

            t = Tree([[[]],[]])
            t.signed_antipode() #Same as t.sign().antipode() and t.antipode().sign()
        """
        return self.sign().antipode()

    def __mul__(self, other):
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
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum(( (other,self),  ))
        elif isinstance(other, Tree):
            out = Forest((self, other))
        elif isinstance(other, Forest):
            out = Forest((self,) + other.tree_list)
        elif isinstance(other, ForestSum):
            out = ForestSum(tuple((c, self * f) for c,f in other.term_list))
        else:
            raise ValueError("Cannot multiply Tree by object of type " + str(type(other)))

        return out.reduce()

    __rmul__ = __mul__

    def __pow__(self, n):
        """
        Returns the :math:`n^{th}` power of a tree for a positive integer :math:`n`, given by a forest with :math:`n` copies of the tree.

        :param n: Exponent, a positive integer

        Example usage::

            t = Tree([[]]) ** 3
        """
        if not isinstance(n, int):
            raise ValueError("Exponent in Tree.__pow__ must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise Tree to a negative power")
        if n == 0:
            return EMPTY_TREE

        out = Forest((self,) * n)
        return out.reduce()

    def __add__(self, other):
        """
        Adds a tree to a scalar, Tree, Forest or ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 + Tree([[]]) + Forest([Tree([]), Tree([[],[]])])
        """
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum((  (1, self), (other, EMPTY_FOREST)  ))
        elif isinstance(other, Tree) or isinstance(other, Forest):
            out = ForestSum((  (1, self), (1, other)  ))
        elif isinstance(other, ForestSum):
            out = ForestSum( ((1, self),) + other.term_list )
        else:
            raise ValueError("Cannot add Tree and " + str(type(other)))

        return out.reduce()

    def __sub__(self, other):
        return self + (-other)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        temp = self * (-1)
        return temp

    def __eq__(self, other):
        """
        Compares the tree with another object and returns true if they represent the same tree, regardless of class type
        (Tree, Forest or ForestSum) or possible reorderings of the same tree.

        :param other: Tree, Forest or ForestSum
        :rtype: bool

        Example usage::

            Tree([[],[]]) == Tree([[],[]]).as_forest() #True
            Tree([[],[]]) == Tree([[],[]]).as_forest_sum() #True
            Tree([[[]],[]]) == Tree([[],[[]]]) #True
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.as_forest_sum() == other * EMPTY_TREE
        if isinstance(other, Tree):
            return self._equals(other)
        elif isinstance(other, Forest):
            return self.as_forest() == other
        elif isinstance(other, ForestSum):
            return self.as_forest_sum() == other
        else:
            raise ValueError("Cannot check equality of Tree and " + str(type(other)))

    def sorted_list_repr(self):
        """
        Returns the list representation of the sorted tree, where the heaviest branches are rotated to the left.

        :return: Sorted list representation
        :rtype: list

        Example usage::

            t = Tree([[],[[]]])
            t.sorted_list_repr() #Returns [[[]],[]]
        """
        return _sorted_list_repr(self.list_repr)

    def level_sequence(self):
        """
        Returns the level sequence of the tree, defined as the list :math:`{\\ell_1, \\ell_2, \\cdots, \\ell_n}`, where
        :math:`\\ell_i` is the level of the :math:`i^{th}` node when the nodes are ordered lexicographically.

        :return: Level sequence
        :rtype: list

        Example usage::

            t = Tree([[[]],[]])
            t.level_sequence() #Returns [0, 1, 2, 1]
        """
        return _list_repr_to_level_sequence(self.list_repr)

    def sorted(self):
        """
        Returns the sorted tree, where the heaviest branches are rotated to the left.

        :return: Sorted tree
        :rtype: Tree

        Example usage::

            t = Tree([[],[[]]])
            t.level_sequence() #Returns Tree([[[]],[]])
        """
        return Tree(self.sorted_list_repr())

    def _equals(self, other_tree):
        return self.sorted_list_repr() == other_tree.sorted_list_repr()

    def as_forest(self):
        """
        Returns the tree t as a forest. Equivalent to Forest([t]).

        :return: Tree as a forest
        :rtype: Forest

        Example usage::

            t = Tree([[],[[]]])
            t.as_forest() #Returns Forest([Tree([[[]],[]])])
        """
        return Forest((self,))

    def as_forest_sum(self):
        """
        Returns the tree t as a forest sum. Equivalent to ForestSum([Forest([t])]).

        :return: Tree as a forest sum
        :rtype: ForestSum

        Example usage::

            t = Tree([[],[[]]])
            t.as_forest_sum() #Returns ForestSum([Forest([Tree([[[]],[]])])])
        """
        return ForestSum(( (1, self), ))

    def apply(self, func):
        """
        Apply a function defined on trees.

        :param func: A function defined on trees
        :type func: callable
        :return: Value of func on the tree

        Example usage::

            func = lambda x : 1. / x.factorial()

            t = Tree([[],[[]]])
            t.apply(func) #Returns 1/8
        """
        return func(self)

    @cache
    def apply_power(self, func, n):
        """
        Apply the power of a function defined on trees, where the product of functions is defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        and negative powers are defined as :math:`f^{-n} = f^n \\circ S`, where :math:`S` is the antipode.

        :param func: A function defined on trees
        :type func: callable
        :param n: Exponent
        :type n: int
        :return: Value of func^n on the tree

        Example usage::

            func = lambda x : 1. / x.factorial()

            t = Tree([[],[[]]])
            t.apply(func, 3)
        """
        res = None
        if n == 0:
            res = self.apply(_counit)
        elif n == 1:
            res = self.apply(func)
        elif n < 0:
            res = self.antipode().apply_power(func, -n)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1))

        if not (isinstance(res, int) or isinstance(res, float)):
            res = res.reduce()
        return res

    @cache
    def apply_product(self, func1, func2):
        """
        Apply the product of two functions, defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :return: Value of the product of functions evaluated on the tree

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            t = Tree([[],[[]]])
            t.apply_product(func1, func2) #Returns t
        """
        subtrees, branches = self.coproduct()
        #a(branches) * b(subtrees)
        if len(subtrees) == 0:
            return 0
        out = branches[0].apply(func1) * subtrees[0].apply(func2)
        for i in range(1, len(subtrees)):
            out += branches[i].apply(func1) * subtrees[i].apply(func2)

        if not (isinstance(out, int) or isinstance(out, float)):
            out = out.reduce()

        return out

    @cache
    def apply_cem_product(self, func1, func2):
        """
        Apply the Calaque, Ebrahimi-Fard and Manchon product of two functions, defined by

        .. math::

            (f \\star g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{CEM} (t).

        For the definition of :math:`\\Delta_{CEM}`, see the documentation for `Tree.cem_coproduct()`.

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :return: Value of the product of functions evaluated on the tree

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            t = Tree([[],[[]]])
            t.apply_cem_product(func1, func2)
        """
        if self.list_repr is None:
            return func2(EMPTY_TREE)

        subtrees, branches = self.cem_coproduct()
        # a(branches) * b(subtrees)
        if len(subtrees) == 0:
            raise
        out = branches[0].apply(func1) * subtrees[0].apply(func2)
        for i in range(1, len(subtrees)):
            out += branches[i].apply(func1) * subtrees[i].apply(func2)

        if not (isinstance(out, int) or isinstance(out, float)):
            out = out.singleton_reduced().reduce()
        return out

    def modified_equation_term(self):
        """
        Returns a forest sum representing a term of the modified equation used in backward error analysis.\n\n

        As described in :cite:`chartier2010algebraic`, given a B-series method :math:`\\Phi_h(y) = B(\\phi, hf, y)`,
        the modified differential equation is a B-series vector field :math:`hf_h(y) = B(\\widetilde{\\phi}, hf, y)` defined
        by

        .. math::

            (\\widetilde{\\phi} \\star e)(t) = \\phi(t)

        where :math:`e(t) = 1 / t!` is the elementary weights function of the exact solution, or equivalently

        .. math::

            \\widetilde{\\phi}(t) = (\\phi \\star e^{\\star (-1)})(t) = \\phi( (\\mathrm{Id} \\star e^{\\star (-1)})(t))

        where :math:`\\mathrm{Id}` is the identity map on trees and :math:`e^{\\star (-1)} = e \\circ S_{CEM}`. This function
        returns :math:`(\\mathrm{Id} \\star e^{\\star (-1)})(t)`, such that applying a map :math:`\\phi` to the result of this
        function returns :math:`\\widetilde{\\phi}`.

        :return: :math:`(\\mathrm{Id} \\star e^{\\star (-1)})(t)`
        """
        ident_ = lambda x : x
        exact_weights_inverse_ = lambda x: x.cem_antipode().apply(lambda x : 1. / x.factorial())
        return self.apply_cem_product(ident_, exact_weights_inverse_)

    def preprocessed_integrator_term(self):
        #TODO
        exact_weights_ = lambda x : 1. / x.factorial()
        ident_inverse_ = lambda x : x.cem_antipode()
        return self.apply_cem_product(exact_weights_, ident_inverse_)

    def __next__(self):
        """
        Generates the next tree with respect to the lexicographic order.

        :return: Next tree
        :rtype: Tree

        Example usage::

                t = Tree([[],[]])
                next(t) # returns Tree([[[[]]]])
        """
        if self.list_repr == None:
            return Tree([])

        layout = self.level_sequence()
        next = _next_layout(layout)
        return Tree(_level_sequence_to_list_repr(next))


######################################
@dataclass(frozen=True)
class Forest():
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
    tree_list : Union[tuple, list]

    def __post_init__(self):
        tuple_repr = tuple(self.tree_list)
        object.__setattr__(self, 'tree_list', tuple_repr)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        memodict[id(self)] = self
        return self

    def __hash__(self):
        counts = Counter(self.reduce().tree_list)
        return hash(frozenset(counts.items()))
    
    def reduce(self):  # Remove redundant empty trees
        """
        Simplify the forest by removing redundant empty trees.

        :return: self
        :rtype: Forest

        Example usage::

            f = Tree([[],[[]]]) * Tree(None)
            f.reduce() #Returns Tree([[],[[]]])
        """
        if len(self.tree_list) <= 1:
            return self

        filtered = tuple(t for t in self.tree_list if t.list_repr is not None)

        if not filtered:
            return EMPTY_FOREST
        elif len(filtered) == len(self.tree_list):
            return self
        else:
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
        for t in self.tree_list:
            yield t
    
    def join(self):
        """
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns the tree :math:`[t_1, t_2, \\cdots, t_k]`.

        :return: :math:`[t_1, t_2, \\cdots, t_k]`
        :rtype: Tree

        Example usage::

            f = Tree([]) * Tree([[]])
            f.join() #Returns Tree([[],[[]]])
        """
        out = [t.list_repr for t in self.tree_list]
        out = tuple(filter(lambda x: x is not None, out))
        return Tree(out)

    
    def nodes(self):
        """
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns the number of nodes in the forest, :math:`\\sum_{i=1}^k |t_i|`.

        :return: Number of nodes
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]])
            f.nodes() #Returns 3
        """
        return sum(t.nodes() for t in self.tree_list)

    def num_trees(self):
        """
        For a forest :math:`t_1 t_2 \\cdots t_k`, returns the number of trees in the forest, :math:`k`.

        :return: Number of trees
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]])
            f.len() #Returns 2
        """
        return len(self.tree_list)

    
    def factorial(self):
        """
        Apply the tree factorial to the forest as a multiplicative map. For a forest :math:`t_1 t_2 \\cdots t_k`,
        returns :math:`\\prod_{i=1}^k t_i!`.

        :return: :math:`\\prod_{i=1}^k t_i!`
        :rtype: int

        Example usage::

            f = Tree([[]]) * Tree([[],[]])
            f.factorial() #Returns 6
        """
        return self.apply(lambda x : x.factorial())

    
    def antipode(self):
        """
        Apply the antipode to the forest as a multiplicative map. For a forest :math:`t_1 t_2 \\cdots t_k`,
        returns :math:`\\prod_{i=1}^k S(t_i)`.

        :return: Antipode
        :rtype: ForestSum

        Example usage::

            f = Tree([[]]) * Tree([[],[]])
            f.antipode()
        """
        if self.tree_list is None or self.tree_list == tuple():
            raise ValueError("Forest is misspecified in Forest.antipode(): self.tree_list is empty")
        elif len(self.tree_list) == 1 and self.tree_list[0]._equals(SINGLETON_TREE):
            return -SINGLETON_FOREST_SUM

        out = self.tree_list[0].antipode()
        for i in range(1, len(self.tree_list)):
            out *= self.tree_list[i].antipode()

        return out.reduce()

    def cem_antipode(self):
        # TODO
        if self.tree_list is None or self.tree_list == tuple():
            raise ValueError("Forest is misspecified in Forest.cem_antipode(): self.tree_list is empty")
        elif len(self.tree_list) == 1 and self.tree_list[0]._equals(SINGLETON_TREE):
            return SINGLETON_FOREST_SUM

        out = self.tree_list[0].cem_antipode()
        for i in range(1, len(self.tree_list)):
            out *= self.tree_list[i].cem_antipode()

        return out.reduce()

    def sign(self):
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
        return self if self.nodes() % 2 == 0 else -self

    def signed_antipode(self):
        """
        Returns the antipode of the signed forest, :math:`S((-1)^{|f|} f)`.

        :return: Antipode of the signed forest, :math:`S((-1)^{|f|} f)`
        :rtype: ForestSum

        .. note::
            Since the antipode and sign functions commute, this function is equivalent to both ``self.sign().antipode()`` and
            ``self.antipode().sign()``.

        Example usage::

            f = Tree([[[]],[]]) * Tree([[]])
            f.signed_antipode() #Same as f.sign().antipode() and f.antipode().sign()
        """
        return self.sign().antipode()

    def __mul__(self, other):
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
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum(( (other, self), ))
        elif isinstance(other, Tree):
            out = Forest(self.tree_list + (other,))
        elif isinstance(other, Forest):
            out = Forest(self.tree_list + other.tree_list)
        elif isinstance(other, ForestSum):
            out = ForestSum(tuple( (c, self * f) for c, f in other.term_list ))
        else:
            raise ValueError("Cannot multiply Forest and " + str(type(other)))

        return out.reduce()

    __rmul__ = __mul__

    def __pow__(self, n):
        """
        Returns the :math:`n^{th}` power of a forest for a positive integer :math:`n`, given by a forest with :math:`n`
        copies of the original forest.

        :param n: Exponent, a positive integer

        Example usage::

            t = ( Tree([]) * Tree([[]]) ) ** 3
        """
        if not isinstance(n, int):
            raise ValueError("Exponent in Forest.__pow__ must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise Forest to a negative power")
        if n == 0:
            return EMPTY_FOREST
        out = Forest(self.tree_list * n)
        return out.reduce()

    def __add__(self, other):
        """
        Adds a forest to a scalar, Tree, Forest or ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 + Tree([[]]) + Forest([Tree([]), Tree([[],[]])])
        """
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum((  (1, self), (other, EMPTY_FOREST)  ))
        elif isinstance(other, Tree) or isinstance(other, Forest):
            out = ForestSum(( (1, self), (1, other) ))
        elif isinstance(other, ForestSum):
            out = ForestSum( ((1, self),) + other.term_list )
        else:
            raise ValueError("Cannot add Forest and " + str(type(other)))

        return out.reduce()

    def __sub__(self, other):
        return self + (-other)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self * (-1)

    def _equals(self, other_forest):
        return Counter(self.reduce().tree_list) == Counter(other_forest.reduce().tree_list)

    def __eq__(self, other):
        """
        Compares the forest with another object and returns true if they represent the same forest, regardless of class type
        (Forest or ForestSum) or possible reorderings of trees.

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
        if isinstance(other, int) or isinstance(other, float):
            return self.as_forest_sum() == other * EMPTY_TREE
        if isinstance(other, Tree):
            return self._equals(other.as_forest())
        elif isinstance(other, Forest):
            return self._equals(other)
        elif isinstance(other, ForestSum):
            return self.as_forest_sum() == other
        else:
            raise ValueError("Cannot check equality of Forest and " + str(type(other)))

    def as_forest_sum(self):
        """
        Returns the forest f as a forest sum. Equivalent to ``ForestSum([f])``.

        :return: Forest as a forest sum
        :rtype: ForestSum

        Example usage::

            f = Tree([[],[[]]]) * Tree([[]])
            f.as_forest_sum() #Returns ForestSum([t])
        """
        return ForestSum(( (1,self), ))

    
    def apply(self, func):
        """
        Given a function defined on trees, apply it multiplicatively to the forest. For a function :math:`g` and forest
        :math:`t_1 t_2 \\cdots t_k`, returns :math:`\\prod_{i=1}^k g(t_i)`.

        :param func: A function defined on trees
        :type func: callable
        :return: Value of func on the forest

        Example usage::

            func = lambda x : 1. / x.factorial()

            f = Tree([[],[[]]]) * Tree([[]])
            f.apply(func) #Returns 1/16
        """
        out = 1
        for t in self.tree_list:
            out = out * t.apply(func)

        if not (isinstance(out, int) or isinstance(out, float) or isinstance(out, Tree)):
            out = out.reduce()
        return out

    def apply_power(self, func, n):
        """
        Apply the power of a function defined on trees, where the product of functions is defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        and negative powers are defined as :math:`f^{-n} = f^n \\circ S`, where :math:`S` is the antipode. Extended multiplicatively to forests.

        :param func: A function defined on trees
        :type func: callable
        :param n: Exponent
        :type n: int
        :return: Value of func^n on the forest

        Example usage::

            func = lambda x : 1. / x.factorial()

            f = Tree([[],[[]]]) * Tree([[]])
            f.apply(func, 3)
        """
        return self.apply(lambda x : x.apply_power(func, n))

    
    def apply_product(self, func1, func2):
        """
        Apply the product of two functions, defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        and extended multiplicatively to forests.

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :return: Value of the product of functions evaluated on the forest

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            f = Tree([[],[[]]]) * Tree([[]])
            f.apply(func1, func2) #Returns f
        """
        self.apply(lambda x : x.apply_product(func1, func2))

    def apply_substitution_product(self, func1, func2):
        #TODO
        self.apply(lambda x : x.apply_cem_product(func1, func2))

    def singleton_reduced(self):
        """
        Removes redundant occurrences of the single-node tree in the forest. If the forest contains a tree with more than
        one node, removes all occurences of the single-node tree. Otherwise, returns the single-node tree.

        :return: Singleton-reduced forest

        Example usage::

            f1 = Tree([]) * Tree([[],[]])
            f2 = Tree([]) * Tree([]) * Tree([])

            f1.singleton_reduced() #Returns Tree([[],[]])
            f2.singleton_reduced() #Returns Tree([])
        """
        out = self.reduce()
        if len(out.tree_list) > 1:
            new_tree_list = tuple(filter(lambda x: x.list_repr != tuple(), out.tree_list))
            if len(new_tree_list) == 0:
                new_tree_list = (SINGLETON_TREE,)
            out = Forest(new_tree_list)
        return out


######################################
@dataclass(frozen=True)
class ForestSum():
    """
    A linear combination of forests.

    :param term_list: A list or tuple containing tuples of coefficients and forests representing terms of the sum.
     If a term contains a tree, it will be converted to a forest on initialisation.

    Example usage::

            t1 = Tree([])
            t2 = Tree([[]])
            t3 = Tree([[[]],[]])

            s = ForestSum([(1, t1), (-2, t1*t2), (1, t2*t3)])
            s == t1 - 2 * t1 * t2 + t2 * t3 #True
    """
######################################
    term_list : Union[tuple, list]

    def __post_init__(self):
        new_term_list = []

        for i in range(len(self.term_list)):
            if isinstance(self.term_list[i][1], Forest):
                new_term_list.append(self.term_list[i])
            elif isinstance(self.term_list[i][1], Tree):
                new_term_list.append((self.term_list[i][0], self.term_list[i][1].as_forest()))
            else:
                raise ValueError("Terms must be tuples of type (int | float, Tree | Forest)")

            if not (isinstance(self.term_list[i][0], int) or isinstance(self.term_list[i][0], float)):
                raise ValueError("Terms must be tuples of type (int | float, Tree | Forest)")

        new_term_list = tuple(new_term_list)
        object.__setattr__(self, 'term_list', new_term_list)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        memodict[id(self)] = self
        return self

    def __hash__(self):
        self_reduced = self.reduce()
        return hash(frozenset(Counter(self.term_list).items()))

    def __repr__(self):
        if len(self.term_list) == 0:
            return "0"
        
        r = ""
        for c, f in self.term_list[:-1]:
            r += repr(c) + "*" + repr(f) + " + "
        r += repr(self.term_list[-1][0]) + "*" + repr(self.term_list[-1][1])
        return r

    def __iter__(self):
        for c,f in self.term_list:
            yield c,f

    def nodes(self):
        """
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`, returns the total number of nodes in the
        forest sum, :math:`\\sum_{i=1}^m \\sum_{j=1}^{k_i} |t_{ij}|`.

        :return: Number of nodes
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]]) + 2 * Tree([[],[]])
            f.nodes() #Returns 6
        """
        return sum(f.nodes() for c, f in self.term_list)

    def num_trees(self):
        """
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`, returns the total number of trees in the
        forest sum, :math:`\\sum_{i=1}^m k_i`.

        :return: Number of trees
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]]) + 2 * Tree([[],[]])
            f.num_trees() #Returns 3
        """
        return sum(f.num_trees() for c, f in self.term_list)

    def num_forests(self):
        """
        For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`, returns the total number of forests in the
        forest sum, :math:`m`.

        :return: Number of forests
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]]) + 2 * Tree([[],[]])
            f.num_trees() #Returns 2
        """
        return len(self.term_list)

    
    def reduce(self):
        """
        Simplify the forest sum in-place by removing redundant empty trees and cancelling terms where applicable.

        :return: self
        :rtype: ForestSum

        Example usage::

            s = Tree([[],[[]]]) * Tree(None) + Tree([]) + Tree([[]]) - Tree([[]])
            s.reduce() #Returns Tree([[],[[]]]) + Tree([])
        """
        new_forest_list = []
        new_coeff_list = []

        for c, f in self.term_list:
            f_reduced = f.reduce()

            for i, f2 in enumerate(new_forest_list):
                if f_reduced._equals(f2):
                    new_coeff_list[i] += c
                    break
            else:
                new_forest_list.append(f_reduced)
                new_coeff_list.append(c)

        result = tuple((c, f) for c, f in zip(new_coeff_list, new_forest_list) if c != 0)

        if not result:
            return ZERO_FOREST_SUM
        else:
            return ForestSum(result)

    def factorial(self):
        """
        Apply the tree factorial to the forest sum as a multiplicative linear map. For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}!`.

        :return: :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}!`
        :rtype: int

        Example usage::

            s = Tree([[],[[]]]) * Tree([]) + Tree([[]])
            s.factorial() #Returns 10
        """
        return self.apply(lambda x : x.factorial())

    
    def antipode(self):
        """
        Apply the antipode to the forest sum as a multiplicative linear map. For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} S(t_{ij})`.

        :return: Antipode
        :rtype: ForestSum

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.antipode()
        """
        c, f = self.term_list[0]
        out = c * f.antipode()
        for c, f in self.term_list[1:]:
            out = out + c * f.antipode()

        return out.reduce()

    def cem_antipode(self):
        # TODO
        c, f = self.term_list[0]
        out = c * f.cem_antipode()
        for c, f in self.term_list[1:]:
            out = out + c * f.cem_antipode()

        return out.reduce()
    
    def sign(self):
        """
        Returns the forest sum where every forest is replaced by its signed value, :math:`(-1)^{|f|} f`.

        :return: Signed forest sum
        :rtype: ForestSum

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.sign() #Returns Tree([[[]],[]]) * Tree([[]]) - 2 * Tree([])
        """
        return ForestSum(tuple((-c if f.nodes() % 2 else c, f) for c,f in self.term_list))

    def signed_antipode(self):
        """
        Returns the antipode of the signed forest sum.

        :return: Antipode of the signed forest sum
        :rtype: ForestSum

        .. note::
            Since the antipode and sign functions commute, this function is equivalent to both ``self.sign().antipode()`` and
            ``self.antipode().sign()``.

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.signed_antipode() #Same as s.sign().antipode() and s.antipode().sign()
        """
        return self.sign().antipode()

    def __mul__(self, other):
        """
        Multiplies a ForestSum by a scalar, Tree, Forest or ForestSum, returning a ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 * Tree([[]]) * ForestSum([Tree([]), Tree([[],[]])], [1, -2])
        """
        if isinstance(other, int) or isinstance(other, float):
            new_term_list = tuple( (c * other, f) for c, f in self.term_list )
        elif isinstance(other, Tree) or isinstance(other, Forest):
            new_term_list = tuple( (c, f * other) for c, f in self.term_list )
        elif isinstance(other, ForestSum):
            new_term_list = tuple( (c1 * c2, f1 * f2)  for c1, f1 in self.term_list for c2, f2 in other.term_list)
        else:
            raise ValueError("Cannot multiply ForestSum and " + str(type(object)))

        out = ForestSum(new_term_list)
        return out.reduce()

    __rmul__ = __mul__


    def __pow__(self, n):
        """
        Returns the :math:`n^{th}` power of a forest sum for a positive integer :math:`n`.

        :param n: Exponent, a positive integer
        :rtype: ForestSum

        Example usage::

            t = ( Tree([]) * Tree([[]]) + Tree([[],[]]) ) ** 3
        """
        if not isinstance(n, int):
            raise ValueError("Exponent in ForestSum.__pow__ must be an int, not " + str(type(n)))
        if n < 0:
            raise ValueError("Cannot raise ForestSum to a negative power")
        if n == 0:
            return EMPTY_FOREST_SUM

        temp = self
        for i in range(n-1):
            temp = temp * self

        return temp.reduce()

    def __add__(self, other):
        """
        Adds a ForestSum to a scalar, Tree, Forest or ForestSum.

        :param other: A scalar, Tree, Forest or ForestSum
        :rtype: ForestSum

        Example usage::

            t = 2 + Tree([[]]) + ForestSum([Tree([]), Tree([[],[]])], [1, -2])
        """
        if isinstance(other, int) or isinstance(other, float):
            new_term_list = self.term_list + ((other, EMPTY_FOREST),)
        elif isinstance(other, Tree) or isinstance(other, Forest):
            new_term_list = self.term_list + ((1, other),)
        elif isinstance(other, ForestSum):
            new_term_list = self.term_list + other.term_list
        else:
            raise ValueError("Cannot add ForestSum and " + str(type(other)))

        out = ForestSum(new_term_list)
        return out.reduce()

    def __sub__(self, other):
        return self + (- other)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self * (-1)

    def _equals(self, other):
        self_reduced = self.reduce()
        other_reduced = other.reduce()
        return Counter(self_reduced.term_list) == Counter(other_reduced.term_list)

    
    def __eq__(self, other):
        """
        Compares the forest sum with another forest sum and returns true if they represent the same forest sum,
        regardless of possible reorderings of trees.

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
        if isinstance(other, int) or isinstance(other, float):
            return self._equals(other * EMPTY_TREE)
        if isinstance(other, Tree):
            return self._equals(other.as_forest_sum())
        elif isinstance(other, Forest):
            return self._equals(other.as_forest_sum())
        elif isinstance(other, ForestSum):
            return self._equals(other)
        else:
            raise ValueError("Cannot check equality of ForestSum and " + str(type(other)))

    
    def apply(self, func):
        """
        Given a function defined on trees, apply it as a multiplicative linear map to the forest sum. For a forest sum :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\\sum_{i=1}^m c_i \\prod_{j=1}^{k_i} g(t_{ij})`.

        :param func: A function defined on trees
        :type func: callable
        :return: Value of func on the forest sum

        Example usage::

            func = lambda x : 1. / x.factorial()

            s = Tree([[],[[]]]) * Tree([[]]) + 2 * Tree([])
            s.apply(func)
        """
        out = 0
        for c, f in self.term_list:
            term = 1
            for t in f.tree_list:
                term = term * func(t)
            out += c * term

        if not (isinstance(out, int) or isinstance(out, float) or isinstance(out, Tree)):
            out = out.reduce()
        return out

    def apply_power(self, func, n):
        """
        Apply the power of a function defined on trees, where the product of functions is defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        and negative powers are defined as :math:`f^{-n} = f^n \\circ S`, where :math:`S` is the antipode. Extended to a multiplicative linear map on forest sums.

        :param func: A function defined on trees
        :type func: callable
        :param n: Exponent
        :type n: int
        :return: Value of func^n on the forest sum

        Example usage::

            func = lambda x : 1. / x.factorial()

            s = Tree([[],[[]]]) * Tree([[]]) + 2 * Tree([])
            s.apply(func, 3)
        """
        return self.apply(lambda x : x.apply_power(func, n))

    
    def apply_product(self, func1, func2):
        """
        Apply the product of two functions, defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        and extended to a multiplicative linear map on forest sums.

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :return: Value of the product of functions evaluated on the forest sum

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            s = Tree([[],[[]]]) * Tree([[]]) + 2 * Tree([])
            s.apply(func1, func2) #Returns s
        """
        return self.apply(lambda x : x.apply_product(func1, func2))

    def apply_substitution_product(self, func1, func2):
        #TODO
        return self.apply(lambda x: x.apply_cem_product(func1, func2))

    def singleton_reduced(self):
        """
        Removes redundant occurrences of the single-node tree in each forest of the forest sum. If the forest contains a tree with more than
        one node, removes all occurences of the single-node tree. Otherwise, replaces it with the single-node tree.

        :return: Singleton-reduced forest sum
        :rtype: ForestSum

        Example usage::

            s1 = Tree([]) * Tree([[],[]]) + Tree([]) * Tree([]) * Tree([])
            s1.singleton_reduced() #Returns Tree([[],[]]) + Tree([])
        """
        return ForestSum(tuple((c, f.singleton_reduced()) for c, f in self.term_list))


##############################################
##############################################

def _is_tree_like(obj):
    return isinstance(obj, Tree) or isinstance(obj, Forest) or isinstance(obj, ForestSum)

EMPTY_TREE = Tree(None)
EMPTY_FOREST = Forest((EMPTY_TREE,))
EMPTY_FOREST_SUM = ForestSum( ( (1, EMPTY_FOREST), ) )
ZERO_FOREST_SUM = ForestSum( ( (0, EMPTY_FOREST), ) )

SINGLETON_TREE = Tree(tuple())
SINGLETON_FOREST = Forest((SINGLETON_TREE,))
SINGLETON_FOREST_SUM = ForestSum( ( (1, SINGLETON_FOREST), ) )