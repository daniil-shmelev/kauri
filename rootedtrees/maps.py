from .trees import Tree, Forest, ForestSum
from .gentrees import trees_of_order
import copy

class Map:
    def __init__(self, func):
        self.func = func

    def __call__(self, t):
        return t.apply(self.func)

    def inverse(self):
        return Map(lambda x : self(x.antipode()))

    def __pow__(self, n):
        if not isinstance(n, int):
            raise ValueError("Map.__pow__ received invalid exponent")

        return Map(lambda x : x.apply_power(self.func, n))

    def __imul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.func = lambda x : other * self.func(x)
        elif isinstance(other, Map):
            self.func = lambda x: x.apply_product(self.func, other.func)
        else:
            raise

    def __mul__(self, other):
        temp = copy.deepcopy(self)
        temp *= other
        return temp

    def __iadd__(self, other):
        if isinstance(other, Map):
            self.func = lambda x: self.func(x) + other.func(x)
        else:
            raise

    def __add__(self, other):
        temp = copy.deepcopy(self)
        temp += other
        return temp

    def __neg__(self):
        return Map(lambda x : -self.func(x))

    def __isub__(self, other):
        self.__iadd__(-other)

    def __sub__(self, other):
        temp = copy.deepcopy(self)
        temp -= other
        return temp

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__


# These are just convenience maps to be used with "apply" and "apply_product"

ident = Map(lambda x : x)
counit = Map(lambda x : 1 if x == Tree(None) else 0)
S = Map(lambda x : x.antipode())
exact_weights = Map(lambda x : 1. / x.factorial())