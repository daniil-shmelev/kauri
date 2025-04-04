import itertools
import copy
from .utils import _nodes, _factorial, _sorted_list_repr, _list_repr_to_level_sequence, _add, _mul, _sub

######################################
class Tree():
######################################
    def __init__(self, list_repr):
        self.list_repr = list_repr

    def __repr__(self):
        return repr(self.list_repr) if self.list_repr is not None else "\u2205"

    def unjoin(self):
        """
        For a tree :math:`t = [t_1, t_2, ..., t_k]`, returns the forest :math:`t_1 t_2 \cdots t_k`

        :return: :math:`t_1 t_2 \cdots t_k`
        :rtype: Forest

        Example usage::

            t = Tree([[[]],[]])
            t.unjoin() #Returns Tree([[]]) * Tree([])
        """
        if self.list_repr is None:
            return Tree(None).as_forest()
        return Forest([Tree(rep) for rep in self.list_repr])


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

    def split(self):
        """
        Returns the coproduct of a tree,

        .. math::

            \Delta(t) = \sum P_c(t) \otimes R_c(t)

        :return: Values of :math:`R_c(t)`
        :rtype: list
        :return: Values of :math:`P_c(t)`
        :rtype: list

        Example usage::

            t = Tree([[[]],[]])
            t.split()
        """
        if self.list_repr == None:
            return [Tree(None)], [Forest([Tree(None)])]
        if self.list_repr == []:
            return [Tree([]), Tree(None)], [Forest([Tree(None)]), Forest([Tree([])])]

        tree_list = []
        forest_list = []
        for rep in self.list_repr:
            t = Tree(rep)
            s, b = t.split()
            tree_list.append(s)
            forest_list.append(b)

        new_tree_list = [Tree(None)]
        new_forest_list = [Forest([self])]

        for p in itertools.product(*tree_list):
            new_tree_list.append(Tree([t.list_repr for t in p if t.list_repr is not None]))

        for p in itertools.product(*forest_list):
            t = []
            for f in p:
                t += f.tree_list
            new_forest_list.append(Forest(t))

        return new_tree_list, new_forest_list

    def antipode(self, apply_reduction = True):
        """
        Returns the antipode of a tree,

        .. math::

            S(t) = -t-\sum S(P_c(t)) R_c(t)

        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Antipode, :math:`S(t)`
        :rtype: ForestSum

        Example usage::

            t = Tree([[[]],[]])
            t.antipode()
        """
        if self.list_repr is None:
            return Tree(None).as_forest_sum()
        elif self.list_repr == []:
            return Tree([]).as_forest_sum().__imul__(-1, False)
        
        subtrees, branches = self.split()
        out = -self.as_forest_sum()
        for i in range(len(subtrees)):
            if subtrees[i]._equals(self) or subtrees[i]._equals(Tree(None)):
                continue
            #out -= branches[i].antipode() * subtrees[i]
            out.__isub__(
                branches[i].antipode(False).__imul__(subtrees[i], False)
                  , False)

        if apply_reduction:
            out.reduce()
        return out

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
            Since the antipode and sign functions commute, this function is equivalent to both self.sign().antipode() and
            self.antipode().sign().

        Example usage::

            t = Tree([[[]],[]])
            t.signed_antipode() #Same as t.sign().antipode() and t.antipode().sign()
        """
        return self.sign().antipode()

    def __mul__(self, other, apply_reduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([Forest([self])], [other])
        elif isinstance(other, Tree):
            out = Forest([self, other])
        elif isinstance(other, Forest):
            out = Forest([self] + other.tree_list)
        elif isinstance(other, ForestSum):
            c = copy.copy(other.coeff_list)
            f = [self * x for x in other.forest_list]
            out = ForestSum(f, c)
        else:
            raise ValueError("oh no")

        if apply_reduction:
            out.reduce()
        return out

    __rmul__ = __mul__

    def __add__(self, other, apply_reduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([Forest([self]), Forest([Tree(None)])], [1, other])
        elif isinstance(other, Tree):
            out = ForestSum([Forest([self]), Forest([other])])
        elif isinstance(other, Forest):
            out = ForestSum([Forest([self]), other])
        elif isinstance(other, ForestSum):
            out = ForestSum([Forest([self])] + other.forest_list, [1] + other.coeff_list)
        else:
            raise ValueError("oh no")

        if apply_reduction:
            out.reduce()
        return out

    def __sub__(self, other, apply_reduction = True):
        return self.__add__(-other, apply_reduction)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self.__mul__(-1, False)

    def __eq__(self, other):
        return self.as_forest_sum() == other

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
        Returns the level sequence of the tree, defined as the list :math:`{\ell_1, \ell_2, \cdots, \ell_n}`, where
        :math:`\ell_i` is the level of the :math:`i^{th}` node when the nodes are ordered lexicographically.

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
        return Forest([self])

    def as_forest_sum(self):
        """
        Returns the tree t as a forest sum. Equivalent to ForestSum([Forest([t])]).

        :return: Tree as a forest sum
        :rtype: ForestSum

        Example usage::

            t = Tree([[],[[]]])
            t.as_forest_sum() #Returns ForestSum([Forest([Tree([[[]],[]])])])
        """
        return ForestSum([Forest([self])])

    def apply(self, func, apply_reduction = True):
        """
        Apply a function defined on trees.

        :param func: A function defined on trees
        :type func: callable
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of func on the tree

        Example usage::

            func = lambda x : 1. / x.factorial()

            t = Tree([[],[[]]])
            t.apply(func) #Returns 1/8
        """
        return func(self)

    def apply_power(self, func, n, apply_reduction = True):
        """
        Apply the power of a function defined on trees, where the product of functions is defined by

        .. math::

            (f \cdot g)(t) := \mu \circ (f \otimes g) \circ \Delta (t)

        :param func: A function defined on trees
        :type func: callable
        :param n: Exponent
        :type n: int
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of func^n on the tree

        Example usage::

            func = lambda x : 1. / x.factorial()

            t = Tree([[],[[]]])
            t.apply(func, 3)
        """
        if n == 0:
            return self
        if n == 1:
            return self.apply(func, apply_reduction)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1, False), False)
            if apply_reduction and not (isinstance(res, int) or isinstance(res, float) or isinstance(res, Tree)):
                res.reduce()
            return res

    
    def apply_product(self, func1, func2, apply_reduction = True):
        """
        Apply the product of two functions, defined by

        .. math::

            (f \cdot g)(t) := \mu \circ (f \otimes g) \circ \Delta (t)

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of the product of functions evaluated on the tree

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            t = Tree([[],[[]]])
            t.apply(func1, func2) #Returns t
        """
        subtrees, branches = self.split()
        #a(branches) * b(subtrees)
        if len(subtrees) == 0:
            return 0
        # out = branches[0].apply(func1) * subtrees[0].apply(func2)
        out = _mul(branches[0].apply(func1), subtrees[0].apply(func2), False)
        for i in range(1, len(subtrees)):
            #out += branches[i].apply(func1) * subtrees[i].apply(func2)
            out = _add(out, _mul(branches[i].apply(func1), subtrees[i].apply(func2), False), False)

        if apply_reduction and not (isinstance(out, int) or isinstance(out, float)):
            out.reduce()
        return out


######################################
class Forest():
######################################
    def __init__(self, tree_list):
        self.tree_list = tree_list
        self.reduce()

    
    def reduce(self):  # Remove redundant empty trees
        """
        Simplify the forest in-place by removing redundant empty trees.

        :return: self
        :rtype: Forest

        Example usage::

            f = Tree([[],[[]]]) * Tree(None)
            f.reduce() #Returns Tree([[],[[]]])
        """
        if len(self.tree_list) > 1:
            self.tree_list = list(filter(lambda x: x.list_repr is not None, self.tree_list))
            if len(self.tree_list) == 0:
                self.tree_list = [Tree(None)]
        return self

    def __repr__(self):
        if len(self.tree_list) == 0:
            return "\u2205"
        
        r = ""
        for t in self.tree_list[:-1]:
            r += repr(t) + " "
        r += repr(self.tree_list[-1]) + ""
        return r

    
    def join(self):
        """
        For a forest :math:`t_1 t_2 \cdots t_k`, returns the tree :math:`[t_1, t_2, \cdots, t_k]`.

        :return: :math:`[t_1, t_2, \cdots, t_k]`
        :rtype: Tree

        Example usage::

            f = Tree([]) * Tree([[]])
            f.join() #Returns Tree([[],[[]]])
        """
        out = [t.list_repr for t in self.tree_list]
        out = list(filter(lambda x: x is not None, out)) 
        return Tree(out)

    
    def nodes(self):
        """
        For a forest :math:`t_1 t_2 \cdots t_k`, returns the number of nodes in the forest, :math:`\sum_{i=1}^k |t_i|`.

        :return: Number of nodes
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]])
            f.nodes() #Returns 3
        """
        return sum(t.nodes() for t in self.tree_list)

    def len(self):
        """
        For a forest :math:`t_1 t_2 \cdots t_k`, returns the number of trees in the forest, :math:`k`.

        :return: Number of trees
        :rtype: int

        Example usage::

            f = Tree([]) * Tree([[]])
            f.len() #Returns 2
        """
        return len(self.tree_list)

    
    def factorial(self):
        """
        Apply the tree factorial to the forest as a multiplicative map. For a forest :math:`t_1 t_2 \cdots t_k`,
        returns :math:`\prod_{i=1}^k t_i!`.

        :return: :math:`\prod_{i=1}^k t_i!`
        :rtype: int

        Example usage::

            f = Tree([[]]) * Tree([[],[]])
            f.factorial() #Returns 6
        """
        return self.apply(lambda x : x.factorial())

    
    def antipode(self, apply_reduction = True):
        """
        Apply the antipode to the forest as a multiplicative map. For a forest :math:`t_1 t_2 \cdots t_k`,
        returns :math:`\prod_{i=1}^k S(t_i)`.

        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Antipode
        :rtype: ForestSum

        Example usage::

            f = Tree([[]]) * Tree([[],[]])
            f.antipode()
        """
        if self.tree_list is None or self.tree_list == []:
            raise ValueError("Forest antipode received empty tree list")
        elif len(self.tree_list) == 1 and self.tree_list[0]._equals(Tree([])):
            return -Tree([])
        
        out = self.tree_list[0].antipode(False)
        for i in range(1, len(self.tree_list)):
            #out *= self.treeList[i].antipode()
            out.__imul__(self.tree_list[i].antipode(False), False)

        if apply_reduction:
            out.reduce()
        return out

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
            Since the antipode and sign functions commute, this function is equivalent to both self.sign().antipode() and
            self.antipode().sign().

        Example usage::

            f = Tree([[[]],[]]) * Tree([[]])
            f.signed_antipode() #Same as f.sign().antipode() and f.antipode().sign()
        """
        return self.sign().antipode()

    def __mul__(self, other, apply_reduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([self], [other])
        elif isinstance(other, Tree):
            out = Forest(self.tree_list + [other])
        elif isinstance(other, Forest):
            out = Forest(self.tree_list + other.tree_list)
        elif isinstance(other, ForestSum):
            f = copy.deepcopy(other.forest_list)
            c = copy.deepcopy(other.coeff_list)
            f = [self * x for x in f]
            out = ForestSum(f, c)
        else:
            raise ValueError("oh no")

        if apply_reduction:
            out.reduce()
        return out

    __rmul__ = __mul__

    
    def __imul__(self, other):
        if isinstance(other, Tree):
            self.tree_list.append(other)
            return self
        if isinstance(other, Forest):
            self.tree_list += other.tree_list
            return self

    def __add__(self, other, apply_reduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([self, Forest([Tree(None)])], [1, other])
        elif isinstance(other, Tree):
            out = ForestSum([self, Forest([other])])
        elif isinstance(other, Forest):
            out = ForestSum([self, other])
        elif isinstance(other, ForestSum):
            out = ForestSum([self] + other.forest_list, [1] + other.coeff_list)
        else:
            raise ValueError("oh no")

        if apply_reduction:
            out.reduce()
        return out

    def __sub__(self, other, apply_reduction = True):
        return self.__add__(-other, apply_reduction)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self.__mul__(-1, False)

    
    def _equals(self, other_forest):
        l1 = self.tree_list
        l2 = copy.copy(other_forest.tree_list)
        for t in l1:
            flag = False
            for i in range(len(l2)):
                if t._equals(l2[i]):
                    l2.pop(i)
                    flag = True
                    break
            if not flag:
                return False
        return len(l2) == 0

    def __eq__(self, other):
        return self.as_forest_sum() == other

    def as_forest_sum(self):
        """
        Returns the forest f as a forest sum. Equivalent to ForestSum([f]).

        :return: Forest as a forest sum
        :rtype: ForestSum

        Example usage::

            f = Tree([[],[[]]]) * Tree([[]])
            f.as_forest_sum() #Returns ForestSum([t])
        """
        return ForestSum([self])

    
    def apply(self, func, apply_reduction = True):
        """
        Given a function defined on trees, apply it multiplicatively to the forest. For a function :math:`g` and forest
        :math:`t_1 t_2 \cdots t_k`, returns :math:`\prod_{i=1}^k g(t_i)`.

        :param func: A function defined on trees
        :type func: callable
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of func on the forest

        Example usage::

            func = lambda x : 1. / x.factorial()

            f = Tree([[],[[]]]) * Tree([[]])
            f.apply(func) #Returns 1/16
        """
        out = 1
        for t in self.tree_list:
            #out = out * t.apply(func)
            out = _mul(out, t.apply(func), False)

        if apply_reduction and not (isinstance(out, int) or isinstance(out, float) or isinstance(out, Tree)):
            out.reduce()
        return out

    def apply_power(self, func, n, apply_reduction = True):
        """
        Apply the power of a function defined on trees, where the product of functions is defined by

        .. math::

            (f \cdot g)(t) := \mu \circ (f \otimes g) \circ \Delta (t)

        and extended multiplicatively to forests.

        :param func: A function defined on trees
        :type func: callable
        :param n: Exponent
        :type n: int
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of func^n on the forest

        Example usage::

            func = lambda x : 1. / x.factorial()

            f = Tree([[],[[]]]) * Tree([[]])
            f.apply(func, 3)
        """
        if n == 0:
            return self
        if n == 1:
            return self.apply(func, apply_reduction)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1, False), False)
            if apply_reduction and not isinstance(res, Tree):
                res.reduce()
            return res

    
    def apply_product(self, func1, func2, apply_reduction = True):
        """
        Apply the product of two functions, defined by

        .. math::

            (f \cdot g)(t) := \mu \circ (f \otimes g) \circ \Delta (t)

        and extended multiplicatively to forests.

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of the product of functions evaluated on the forest

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            f = Tree([[],[[]]]) * Tree([[]])
            f.apply(func1, func2) #Returns f
        """
        out = 1
        for t in self.tree_list:
            out = out * t.apply_product(func1, func2)

        if apply_reduction and not (isinstance(out, int) or isinstance(out, float)):
            out.reduce()
        return out

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
        out = copy.copy(self)
        out.reduce()
        if len(out.tree_list) > 1:
            out.tree_list = list(filter(lambda x: x.list_repr != [], out.tree_list))
            if len(out.tree_list) == 0:
                out.tree_list = [Tree([])]
        return out


######################################
class ForestSum():
######################################
    def __init__(self, forest_list, coeff_list = None):
        self.forest_list = forest_list
        if coeff_list == None:
            self.coeff_list = [1] * len(forest_list)
        else:
            if len(coeff_list) != len(forest_list):
                raise ValueError("forest list and coefficient list are of different lengths")
            else:
                self.coeff_list = coeff_list

        self.reduce()

    def __repr__(self):
        if len(self.forest_list) == 0:
            return "0"
        
        r = ""
        for f, c in zip(self.forest_list[:-1], self.coeff_list[:-1]):
            r += repr(c) + "*" + repr(f) + " + "
        r += repr(self.coeff_list[-1]) + "*" + repr(self.forest_list[-1])
        return r

    
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
        for f, c in zip(self.forest_list, self.coeff_list):
            try:
                i = new_forest_list.index(f)
                new_coeff_list[i] += c
            except:
                new_forest_list.append(f)
                new_coeff_list.append(c)

        zero_idx = []
        for i in range(len(new_coeff_list)):
            if new_coeff_list[i] == 0:
                zero_idx.append(i)

        for i in zero_idx[::-1]:
            new_forest_list.pop(i)
            new_coeff_list.pop(i)

        if new_forest_list == []:
            new_forest_list.append(Tree(None).as_forest())
            new_coeff_list.append(0)

        self.forest_list = new_forest_list
        self.coeff_list = new_coeff_list
        return self

    def factorial(self):
        """
        Apply the tree factorial to the forest sum as a multiplicative linear map. For a forest sum :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} t_{ij}!`.

        :return: :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} t_{ij}!`
        :rtype: int

        Example usage::

            s = Tree([[],[[]]]) * Tree([]) + Tree([[]])
            s.factorial() #Returns 10
        """
        return self.apply(lambda x : x.factorial(), False)

    
    def antipode(self, applyReduction = True):
        """
        Apply the antipode to the forest sum as a multiplicative linear map. For a forest sum :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} S(t_{ij})`.

        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Antipode
        :rtype: ForestSum

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.antipode()
        """
        out = self.coeff_list[0] * self.forest_list[0].antipode()
        for i in range(1, len(self.forest_list)):
            #out += self.coeff_list[i] * self.forest_list[i].antipode()
            out.__iadd__(
                _mul(self.coeff_list[i], self.forest_list[i].antipode(False), False)
            , False)

        if applyReduction:
            out.reduce()
        return out

    
    def sign(self):
        """
        Returns the forest sum where every forest is replaced by its signed value, :math:`(-1)^{|f|} f`.

        :return: Signed forest sum
        :rtype: ForestSum

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.sign() #Returns Tree([[[]],[]]) * Tree([[]]) - 2 * Tree([])
        """
        new_coeffs = []
        for i in range(len(self.coeff_list)):
            if self.forest_list[i].nodes() % 2 == 0:
                new_coeffs.append(self.coeff_list[i])
            else:
                new_coeffs.append(-self.coeff_list[i])
        return ForestSum(self.forest_list, new_coeffs)

    def signed_antipode(self):
        """
        Returns the antipode of the signed forest sum.

        :return: Antipode of the signed forest sum
        :rtype: ForestSum

        .. note::
            Since the antipode and sign functions commute, this function is equivalent to both self.sign().antipode() and
            self.antipode().sign().

        Example usage::

            s = Tree([[[]],[]]) * Tree([[]]) + 2 * Tree([])
            s.signed_antipode() #Same as s.sign().antipode() and s.antipode().sign()
        """
        return self.sign().antipode()

    def __imul__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            self.coeff_list = [x * other for x in self.coeff_list]
        elif isinstance(other, Tree) or isinstance(other, Forest):
            self.forest_list = [x * other for x in self.forest_list]
        elif isinstance(other, ForestSum):
            self.forest_list = [x * y for x in self.forest_list for y in other.forest_list]
            self.coeff_list = [x * y for x in self.coeff_list for y in other.coeff_list]
        else:
            raise ValueError("oh no")

        if applyReduction:
            self.reduce()
        return self

    def __mul__(self, other, applyReduction = True):
        temp = copy.deepcopy(self)
        temp.__imul__(other, applyReduction)
        return temp

    __rmul__ = __mul__


    def __iadd__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            self.forest_list += [Forest([Tree(None)])]
            self.coeff_list += [other]
        elif isinstance(other, Tree):
            self.forest_list += [Forest([other])]
            self.coeff_list += [1]
        elif isinstance(other, Forest):
            self.forest_list += [other]
            self.coeff_list += [1]
        elif isinstance(other, ForestSum):
            self.forest_list += other.forest_list
            self.coeff_list += other.coeff_list
        else:
            raise ValueError("oh no")

        if applyReduction:
            self.reduce()
        return self

    def __add__(self, other, applyReduction = True):
        temp = copy.deepcopy(self)
        temp.__iadd__(other, applyReduction)
        return temp

    def __isub__(self, other, applyReduction = True):
        self.__iadd__(-other, applyReduction)
        return self

    def __sub__(self, other, applyReduction = True):
        temp = copy.deepcopy(self)
        temp.__isub__(other, applyReduction)
        return temp

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self.__mul__(-1, False)

    
    def __eq__(self, other):
        temp = copy.copy(other)
        if isinstance(temp, int) or isinstance(temp, float):
            temp = Tree(None).__mul__(temp, False)
        elif not isinstance(temp, ForestSum):
            temp = temp.as_forest_sum()

        f1 = self.forest_list
        c1 = self.coeff_list
        f2 = temp.forest_list
        c2 = temp.coeff_list
        for forest1,coeff1 in zip(f1, c1):
            flag = False
            for i in range(len(f2)):
                if forest1._equals(f2[i]) and coeff1 == c2[i]:
                    f2.pop(i)
                    c2.pop(i)
                    flag = True
                    break
            if not flag:
                return False

        return len(f2) == 0

    
    def apply(self, func, apply_reduction = True):
        """
        Given a function defined on trees, apply it as a multiplicative linear map to the forest sum. For a forest sum :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} t_{ij}`,
        returns :math:`\sum_{i=1}^m c_i \prod_{j=1}^{k_i} g(t_{ij})`.

        :param func: A function defined on trees
        :type func: callable
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of func on the forest sum

        Example usage::

            func = lambda x : 1. / x.factorial()

            s = Tree([[],[[]]]) * Tree([[]]) + 2 * Tree([])
            s.apply(func)
        """
        out = 0
        for f,c in zip(self.forest_list, self.coeff_list):
            term = 1
            for t in f.tree_list:
                #term *= func(t)
                term = _mul(term, func(t), False)
            #out += c * term
            out = _add(out, _mul(term, c, False), False)

        if apply_reduction and not (isinstance(out, int) or isinstance(out, float) or isinstance(out, Tree)):
            out.reduce()
        return out

    def apply_power(self, func, n, apply_reduction = True):
        """
        Apply the power of a function defined on trees, where the product of functions is defined by

        .. math::

            (f \cdot g)(t) := \mu \circ (f \otimes g) \circ \Delta (t)

        and extended to a multiplicative linear map on forest sums.

        :param func: A function defined on trees
        :type func: callable
        :param n: Exponent
        :type n: int
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of func^n on the forest sum

        Example usage::

            func = lambda x : 1. / x.factorial()

            s = Tree([[],[[]]]) * Tree([[]]) + 2 * Tree([])
            s.apply(func, 3)
        """
        if n == 0:
            return self
        if n == 1:
            return self.apply(func, apply_reduction)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1, False), False)
            if apply_reduction and not isinstance(res, Tree):
                res.reduce()
            return res

    
    def apply_product(self, func1, func2, apply_reduction = True):
        """
        Apply the product of two functions, defined by

        .. math::

            (f \cdot g)(t) := \mu \circ (f \otimes g) \circ \Delta (t)

        and extended to a multiplicative linear map on forest sums.

        :param func1: A function defined on trees
        :type func1: callable
        :param func2: A function defined on trees
        :type func2: callable
        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :type apply_reduction: bool
        :return: Value of the product of functions evaluated on the forest sum

        Example usage::

            func1 = lambda x : x
            func2 = lambda x : x.antipode()

            s = Tree([[],[[]]]) * Tree([[]]) + 2 * Tree([])
            s.apply(func1, func2) #Returns s
        """
        out = 0
        for f,c in zip(self.forest_list, self.coeff_list):
            term = 1
            for t in f.tree_list:
                #term *= t.apply_product(func1, func2)
                term = _mul(term, t.apply_product(func1, func2), False)
            #out += c * term
            out = _add(out, _mul(c, term, False), False)

        if apply_reduction and not (isinstance(out, int) or isinstance(out, float)):
            out.reduce()
        return out

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
        return ForestSum([x.singleton_reduced() for x in self.forest_list], self.coeff_list)