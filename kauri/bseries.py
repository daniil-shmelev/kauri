"""
BSeries
"""
import sympy as sp
from kauri import Tree, trees_up_to_order, Map
from functools import cache
import itertools

@cache
def _elementary_differential(tree : Tree, f : sp.ImmutableDenseMatrix, y_vars : sp.ImmutableDenseMatrix):
    if tree.list_repr is None:
        return y_vars # y
    if len(tree.list_repr) == 1:
        return f # f(y)

    # tree = [t_1, ..., t_k], sub_diffs = [F(t_1), ..., F(t_k)]
    sub_diffs = tuple(_elementary_differential(subtree, f, y_vars) for subtree in tree.unjoin())

    # Now compute f^(k) (F(t_1), ..., F(t_k))
    # which equals \sum_{i_j = 1,...,d} F(t_1)_{i_1} ... F(t_k)_{i_k} ( d^k f / dy_{i_1} ... dy_{i_k} )
    result = sp.zeros(*sp.shape(y_vars))
    dim = len(y_vars)
    k = len(tree.list_repr) - 1

    for idx in itertools.product(range(dim), repeat=k):
        # Compute the derivative d^k f / dy_{i_1} ... dy_{i_k} first
        term = f
        for i in idx:
            term = sp.diff(term, y_vars[i])

        # Now multiply by F(t_1)_{i_1} ... F(t_k)_{i_k}
        for j, i in enumerate(idx):
            term *= sub_diffs[j][i]
        result += term

    return result

def elementary_differential(tree : Tree, f : sp.Matrix, y : sp.Matrix) -> sp.Matrix:
    """
    Returns the elementary differential of a vector field.
    These are defined recursively on trees by:

    .. math::

        F(\\emptyset) = y,
    .. math::


        F(\\bullet) = f(y),
    .. math::


        F([t_1, t_2, \\ldots, t_k])(y) = f^{(k)}(y)(F(t_1)(y), F(t_2)(y), \\ldots, F(t_m)(y)).

    :param tree: Tree corresponding to the elementary differential
    :type tree: Tree
    :param f: Vector field
    :type f: sympy.Matrix
    :param y: Symbolic variables y
    :type y: sympy.Matrix

    Example usage::

            import kauri as kr
            import sympy as sp

            y1, y2 = sp.symbols('y1 y2')
            y = sp.Matrix([y1, y2])
            f = sp.Matrix([y1 ** 2, y1 * y2])

            t = kr.Tree([[[]],[]])
            elementary_differential(t, f, y) # Returns sp.Matrix([[4 * y1**5 ], [ 4 * y1**4 * y2]])
    """
    return _elementary_differential(tree, sp.ImmutableDenseMatrix(f), sp.ImmutableDenseMatrix(y))


class BSeries:
    """
    This class allows for the symbolic manipulation and evaluation of truncated
    B-Series on unlabelled trees, for a given vector field f. Given a weights
    function :math:`\\varphi`, the associated truncated B-Series is

    .. math::

        B_h(\\varphi, y_0) := \\sum_{|t| \\leq n} \\frac{h^{|t|}}{\\sigma(t)} \\varphi(t) F(t)(y_0),

    where the sum runs over all trees of order at most :math:`n`.

    :param y: Symbolic variables y
    :type y: sympy.Matrix
    :param f: Vector field
    :type f: sympy.Matrix
    :param weights: The weights function :math:`\\varphi` corresponding to the B-Series.
    :type weights: kauri.Map
    :param order: The truncation order of the B-Series
    :type order: int

    Example usage::

            import kauri as kr
            import sympy as sp

            y1 = sp.symbols('y1')
            y = sp.Matrix([y1])
            f = sp.Matrix([y1 ** 2])

            m = kr.rk4.elementary_weights_map()
            bs = BSeries(y, f, m, 5)

            print(bs.symbolic()) # Print the B-Series as a sympy expression
            print(bs(1, 0.1)) # Evaluate the B-Series at y = 1, h = 0.1
    """

    def __init__(self, y : sp.Matrix, f : sp.Matrix, weights : Map, order : int):
        self.f = sp.ImmutableDenseMatrix(f) #Immutable for cache in elementary_differential
        self.y = sp.ImmutableDenseMatrix(y)
        self.h = sp.symbols('h')
        self.symbolic_expr = sp.zeros(*sp.shape(y))
        for t in trees_up_to_order(order):
            self.symbolic_expr = self.symbolic_expr + self.h ** t.nodes() * weights(t) * _elementary_differential(t, self.f, self.y) / t.sigma()

    def __call__(self, y, h):
        out = self.symbolic_expr.subs(self.h, h)
        if len(self.y) == 1:
            out = out.subs(self.y[0], y)
        else:
            for i in range(len(self.y)):
                out = out.subs(self.y[i], y[i])
        return [float(x) for x in out]

    def symbolic(self):
        pass

    def __and__(self, other): #Composition of B-Series, given by the product of characters
        pass
