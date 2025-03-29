import itertools
import copy
from .utils import *

from fractions import Fraction

useFractionsInRepr_ = False

def useFractions(flag):
    useFractionsInRepr_ = flag

def c_repr(f):
    if useFractionsInRepr_:
        frac = Fraction.from_float(f)
        return "(" + str(frac) + ")"
    else:
        return repr(f)

######################################
class Tree():
######################################
    def __init__(self, listRepr):
        self.listRepr = listRepr

    def __repr__(self):
        return repr(self.listRepr) if self.listRepr is not None else "\u2205"

    def unjoin(self):
        if self.listRepr is None:
            return Tree(None).asForest()
        return Forest([Tree(rep) for rep in self.listRepr])

    def numNodes(self):
        return numNodes_(self.listRepr)

    def factorial(self):
        return factorial_(self.listRepr)[0]

    def split(self):
        if self.listRepr == None:
            return [Tree(None)], [Forest([Tree(None)])]
        if self.listRepr == []:
            return [Tree([]), Tree(None)], [Forest([Tree(None)]), Forest([Tree([])])]

        tree_list = []
        forest_list = []
        for rep in self.listRepr:
            t = Tree(rep)
            s, b = t.split()
            tree_list.append(s)
            forest_list.append(b)

        new_tree_list = [Tree(None)]
        new_forest_list = [Forest([self])]

        for p in itertools.product(*tree_list):
            new_tree_list.append(Tree([t.listRepr for t in p if t.listRepr is not None]))

        for p in itertools.product(*forest_list):
            t = []
            for f in p:
                t += f.treeList
            new_forest_list.append(Forest(t))

        return new_tree_list, new_forest_list

    def antipode(self, applyReduction = True):
        if self.listRepr is None:
            return Tree(None).asForestSum()
        elif self.listRepr == []:
            return Tree([]).asForestSum().__imul__(-1, False)
        
        subtrees, branches = self.split()
        out = -self.asForestSum()
        for i in range(len(subtrees)):
            if subtrees[i].equals(self) or subtrees[i].equals(Tree(None)):
                continue
            #out -= branches[i].antipode() * subtrees[i]
            out.__isub__(
                branches[i].antipode(False).__imul__(subtrees[i], False)
                  , False)

        if applyReduction:
            out.reduce()
        return out

    def sign(self):
        return self if self.numNodes() % 2 == 0 else -self

    def signed_antipode(self):
        return self.sign().antipode()

    def __mul__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([Forest([self])], [other])
        elif isinstance(other, Tree):
            out = Forest([self, other])
        elif isinstance(other, Forest):
            out = Forest([self] + other.treeList)
        elif isinstance(other, ForestSum):
            c = copy.copy(other.coeffList)
            f = [self * x for x in other.forestList]
            out = ForestSum(f, c)
        else:
            raise ValueError("oh no")

        if applyReduction:
            out.reduce()
        return out

    __rmul__ = __mul__

    def __add__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([Forest([self]), Forest([Tree(None)])], [1, other])
        elif isinstance(other, Tree):
            out = ForestSum([Forest([self]), Forest([other])])
        elif isinstance(other, Forest):
            out = ForestSum([Forest([self]), other])
        elif isinstance(other, ForestSum):
            out = ForestSum([Forest([self])] + other.forestList, [1] + other.coeffList)
        else:
            raise ValueError("oh no")

        if applyReduction:
            out.reduce()
        return out

    def __sub__(self, other, applyReduction = True):
        return self.__add__(-other, applyReduction)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self.__mul__(-1, False)

    def __eq__(self, other):
        return self.asForestSum() == other

    def sortedListRepr(self):
        return sortedListRepr_(self.listRepr)

    def layout(self):
        return listReprToLayout_(self.listRepr)

    def sorted(self):
        return Tree(self.sortedListRepr())

    def equals(self, otherTree):
        return self.sortedListRepr() == otherTree.sortedListRepr()

    def asForest(self):
        return Forest([self])

    def asForestSum(self):
        return ForestSum([Forest([self])])

    def apply(self, func, applyReduction = True):
        return func(self)

    def apply_power(self, func, n, applyReduction = True):
        if n == 0:
            return self
        if n == 1:
            return self.apply(func, applyReduction)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1, False), False)
            if applyReduction and not (isinstance(res, int) or isinstance(res, float) or isinstance(res, Tree)):
                res.reduce()
            return res

    
    def apply_product(self, func1, func2, applyReduction = True):
        subtrees, branches = self.split()
        #a(branches) * b(subtrees)
        if len(subtrees) == 0:
            return 0
        # out = branches[0].apply(func1) * subtrees[0].apply(func2)
        out = mul_(branches[0].apply(func1),subtrees[0].apply(func2), False)
        for i in range(1, len(subtrees)):
            #out += branches[i].apply(func1) * subtrees[i].apply(func2)
            out = add_(out, mul_(branches[i].apply(func1), subtrees[i].apply(func2), False), False)

        if applyReduction and not (isinstance(out, int) or isinstance(out, float)):
            out.reduce()
        return out


######################################
class Forest():
######################################
    def __init__(self, treeList):
        self.treeList = treeList
        self.reduce()

    
    def reduce(self):  # Remove redundant empty trees
        if len(self.treeList) > 1:
            self.treeList = list(filter(lambda x: x.listRepr is not None, self.treeList))
            if len(self.treeList) == 0:
                self.treeList = [Tree(None)]
        return self

    def __repr__(self):
        if len(self.treeList) == 0:
            return "\u2205"
        
        r = ""
        for t in self.treeList[:-1]:
            r += repr(t) + " "
        r += repr(self.treeList[-1]) + ""
        return r

    
    def join(self):
        out = [t.listRepr for t in self.treeList]
        out = list(filter(lambda x: x is not None, out)) 
        return Tree(out)

    
    def numNodes(self):
        return sum(t.numNodes() for t in self.treeList)

    def len(self):
        return len(self.treeList)

    
    def factorial(self):
        return self.apply(lambda x : x.factorial())

    
    def antipode(self, applyReduction = True):
        if self.treeList is None or self.treeList == []:
            raise ValueError("Error in forest antipode")
        elif len(self.treeList) == 1 and self.treeList[0].equals(Tree([])):
            return -Tree([])
        
        out = self.treeList[0].antipode(False)
        for i in range(1, len(self.treeList)):
            #out *= self.treeList[i].antipode()
            out.__imul__(self.treeList[i].antipode(False), False)

        if applyReduction:
            out.reduce()
        return out

    def sign(self):
        return self if self.numNodes() % 2 == 0 else -self

    def signed_antipode(self):
        return self.sign().antipode()

    def __mul__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([self], [other])
        elif isinstance(other, Tree):
            out = Forest(self.treeList + [other])
        elif isinstance(other, Forest):
            out = Forest(self.treeList + other.treeList)
        elif isinstance(other, ForestSum):
            f = copy.deepcopy(other.forestList)
            c = copy.deepcopy(other.coeffList)
            f = [self * x for x in f]
            out = ForestSum(f, c)
        else:
            raise ValueError("oh no")

        if applyReduction:
            out.reduce()
        return out

    __rmul__ = __mul__

    
    def __imul__(self, other):
        if isinstance(other, Tree):
            self.treeList.append(other)
            return self
        if isinstance(other, Forest):
            self.treeList += other.treeList
            return self

    def __add__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            out = ForestSum([self, Forest([Tree(None)])], [1, other])
        elif isinstance(other, Tree):
            out = ForestSum([self, Forest([other])])
        elif isinstance(other, Forest):
            out = ForestSum([self, other])
        elif isinstance(other, ForestSum):
            out = ForestSum([self] + other.forestList, [1] + other.coeffList)
        else:
            raise ValueError("oh no")

        if applyReduction:
            out.reduce()
        return out

    def __sub__(self, other, applyReduction = True):
        return self.__add__(-other, applyReduction)

    __radd__ = __add__
    __rsub__ = __sub__

    def __neg__(self):
        return self.__mul__(-1, False)

    
    def equals(self, otherForest):
        l1 = self.treeList
        l2 = copy.copy(otherForest.treeList)
        for t in l1:
            flag = False
            for i in range(len(l2)):
                if t.equals(l2[i]):
                    l2.pop(i)
                    flag = True
                    break
            if not flag:
                return False
        return len(l2) == 0

    def __eq__(self, other):
        return self.asForestSum() == other

    def asForestSum(self):
        return ForestSum([self])

    
    def apply(self, func, applyReduction = True):
        out = 1
        for t in self.treeList:
            #out = out * t.apply(func)
            out = mul_(out, t.apply(func), False)

        if applyReduction and not (isinstance(out, int) or isinstance(out, float) or isinstance(out, Tree)):
            out.reduce()
        return out

    def apply_power(self, func, n, applyReduction = True):
        if n == 0:
            return self
        if n == 1:
            return self.apply(func, applyReduction)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1, False), False)
            if applyReduction and not isinstance(res, Tree):
                res.reduce()
            return res

    
    def apply_product(self, func1, func2, applyReduction = True):
        out = 1
        for t in self.treeList:
            out = out * t.apply_product(func1, func2)

        if applyReduction and not (isinstance(out, int) or isinstance(out, float)):
            out.reduce()
        return out

    def singleton_reduced(self):
        out = copy.copy(self)
        out.reduce()
        if len(out.treeList) > 1:
            out.treeList = list(filter(lambda x: x.listRepr != [], out.treeList))
            if len(out.treeList) == 0:
                out.treeList = [Tree([])]
        return out


######################################
class ForestSum():
######################################
    def __init__(self, forestList, coeffList = None):
        self.forestList = forestList
        if coeffList == None:
            self.coeffList = [1] * len(forestList)
        else:
            if len(coeffList) != len(forestList):
                raise ValueError("len(coeffList) != len(forestList)")
            else:
                self.coeffList = coeffList

        self.reduce()

    def __repr__(self):
        if len(self.forestList) == 0:
            return "0"
        
        r = ""
        for f, c in zip(self.forestList[:-1], self.coeffList[:-1]):
            r += c_repr(c) + "*" + repr(f) + " + "
        r += repr(self.coeffList[-1]) + "*" + repr(self.forestList[-1])
        return r

    
    def reduce(self):
        newForestList = []
        newCoeffList = []
        for f, c in zip(self.forestList, self.coeffList):
            try:
                i = newForestList.index(f)
                newCoeffList[i] += c
            except:
                newForestList.append(f)
                newCoeffList.append(c)

        zero_idx = []
        for i in range(len(newCoeffList)):
            if newCoeffList[i] == 0:
                zero_idx.append(i)

        for i in zero_idx[::-1]:
            newForestList.pop(i)
            newCoeffList.pop(i)

        if newForestList == []:
            newForestList.append(Tree(None).asForest())
            newCoeffList.append(0)

        self.forestList = newForestList
        self.coeffList = newCoeffList
        return self

    def factorial(self):
        return self.apply(lambda x : x.factorial(), False)

    
    def antipode(self, applyReduction = True):
        out = self.coeffList[0] * self.forestList[0].antipode()
        for i in range(1, len(self.forestList)):
            #out += self.coeffList[i] * self.forestList[i].antipode()
            out.__iadd__(
                mul_(self.coeffList[i], self.forestList[i].antipode(False), False)
            , False)

        if applyReduction:
            out.reduce()
        return out

    
    def sign(self):
        newCoeffs = []
        for i in range(len(self.coeffList)):
            if self.forestList[i].numNodes() % 2 == 0:
                newCoeffs.append(self.coeffList[i])
            else:
                newCoeffs.append(-self.coeffList[i])
        return ForestSum(self.forestList, newCoeffs)

    def signed_antipode(self):
        return self.sign().antipode()

    def __imul__(self, other, applyReduction = True):
        if isinstance(other, int) or isinstance(other, float):
            self.coeffList = [x * other for x in self.coeffList]
        elif isinstance(other, Tree) or isinstance(other, Forest):
            self.forestList = [x * other for x in self.forestList]
        elif isinstance(other, ForestSum):
            self.forestList = [x * y for x in self.forestList for y in other.forestList]
            self.coeffList = [x * y for x in self.coeffList for y in other.coeffList]
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
            self.forestList += [Forest([Tree(None)])]
            self.coeffList += [other]
        elif isinstance(other, Tree):
            self.forestList += [Forest([other])]
            self.coeffList += [1]
        elif isinstance(other, Forest):
            self.forestList += [other]
            self.coeffList += [1]
        elif isinstance(other, ForestSum):
            self.forestList += other.forestList
            self.coeffList += other.coeffList
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
            temp = temp.asForestSum()

        f1 = self.forestList
        c1 = self.coeffList
        f2 = temp.forestList
        c2 = temp.coeffList
        for forest1,coeff1 in zip(f1, c1):
            flag = False
            for i in range(len(f2)):
                if forest1.equals(f2[i]) and coeff1 == c2[i]:
                    f2.pop(i)
                    c2.pop(i)
                    flag = True
                    break
            if not flag:
                return False

        return len(f2) == 0

    
    def apply(self, func, applyReduction = True):
        out = 0
        for f,c in zip(self.forestList, self.coeffList):
            term = 1
            for t in f.treeList:
                #term *= func(t)
                term = mul_(term, func(t), False)
            #out += c * term
            out = add_(out, mul_(term, c, False), False)

        if applyReduction and not (isinstance(out, int) or isinstance(out, float) or isinstance(out, Tree)):
            out.reduce()
        return out

    def apply_power(self, func, n, applyReduction = True):
        if n == 0:
            return self
        if n == 1:
            return self.apply(func, applyReduction)
        else:
            res = self.apply_product(func, lambda x : x.apply_power(func, n-1, False), False)
            if applyReduction and not isinstance(res, Tree):
                res.reduce()
            return res

    
    def apply_product(self, func1, func2, applyReduction = True):
        out = 0
        for f,c in zip(self.forestList, self.coeffList):
            term = 1
            for t in f.treeList:
                #term *= t.apply_product(func1, func2)
                term = mul_(term, t.apply_product(func1, func2), False)
            #out += c * term
            out = add_(out, mul_(c, term, False), False)

        if applyReduction and not (isinstance(out, int) or isinstance(out, float)):
            out.reduce()
        return out

    def singleton_reduced(self):
        return ForestSum([x.singleton_reduced() for x in self.forestList], self.coeffList)