"""
This module allows for the symbolic manipulation and evaluation of truncated
B-Series on unlabelled trees, for an ODE of the form

.. math::

    \\frac{dy}{ds} = f(y).

Given a weights function :math:`\\varphi`, the associated truncated B-Series is

.. math::

    B_h(\\varphi, y_0) := \\sum_{|t| \\leq n} \\frac{h^{|t|}}{\\sigma(t)} \\varphi(t) F(t)(y_0),

where the sum runs over all trees of order at most :math:`n`, :math:`\\sigma(t)` is the symmetry factor
of a tree, and :math:`F(t)(y_0)` are the elementary differentials, defined recursively on trees by:

.. math::

    F(\\emptyset) = y,
.. math::

    F(\\bullet) = f(y),
.. math::

    F([t_1, t_2, \\ldots, t_k])(y) = f^{(k)}(y)(F(t_1)(y), F(t_2)(y), \\ldots, F(t_m)(y)).

The main objective of this module is to evaluate existing B-series. For more complicated operations,
it is recommended to work with the underlying character (or elementary weights function) and then
construct a B-series from the result. For example, take the following B-series corresponding to
the Euler and RK4 schemes:

.. code-block:: python

    import kauri as kr

    y1 = sp.symbols('y1')
    y = sp.Matrix([y1])
    f = sp.Matrix([y1 ** 2])

    bs1 = kr.BSeries(y, f, kr.euler.elementary_weights_map(), 5)
    bs2 = kr.BSeries(y, f, kr.rk4.elementary_weights_map(), 5)

At the level of the underlying characters, the composition of two B-series is given by the BCK product
of the characters, so one can get the composition by doing

.. code-block:: python

    bs_composition = kr.BSeries(y, f, bs2.weights * bs1.weights, 5)

Similarly the inverse B-series for the Euler scheme is given by

.. code-block:: python

    bs_inverse = kr.BSeries(y, f, bs1.weights ** (-1), 5)

and the symmetric-adjoint method is given by

.. code-block:: python

    bs_adjoint = kr.BSeries(y, f, bs1.weights ** (-1) & kr.sign, 5)

where `kr.sign` is the :class:`Map` sending `t` to `-t`.

"""
import itertools
from functools import cache
from typing import Union

import sympy as sp

from kauri import Tree, trees_up_to_order, Map

def _check_f_y(f, y):
    # Checks that f and y are correctly specified

    if not isinstance(y, sp.Matrix):
        raise TypeError("y must be of type sympy.Matrix, not " + str(type(y)))
    if not isinstance(f, sp.Matrix):
        raise TypeError("The vector field f must be of type sympy.Matrix, not " + str(type(f)))

    # Make sure f, y are vectors of the same dimensions
    if not (f.shape[1] == 1 and y.shape[1] == 1 and f.shape[0] == y.shape[0]):
        raise ValueError("""f, y must be column vectors, both of shape (d, 1) for some d.
            Instead, got f of shape """ + str(f.shape) + " and y of shape " + str(y.shape))

    # Make sure f is in terms of y and nothing else
    allowed_symbols = set(y)
    f_symbols = set().union(*(expr.free_symbols for expr in f))
    if not f_symbols <= allowed_symbols:
        raise ValueError("""The vector field f contains unknown symbols which are not contained in y.
            If these are constants, please add them to the ODE with a derivative of 0. If these represent
            time, please add them to the ODE with a derivative of 1.""")

@cache
def _elementary_differential(tree : Tree,
                             f : sp.ImmutableDenseMatrix,
                             y_vars : sp.ImmutableDenseMatrix):
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

def elementary_differential(tree : Tree,
                            f : sp.Matrix,
                            y : sp.Matrix
                            ) -> sp.Matrix:
    """
    Returns the elementary differential of a vector field.
    These are defined recursively on trees by:

    .. math::

        F(\\emptyset) = y,
    .. math::


        F(\\bullet) = f(y),
    .. math::


        F([t_1, t_2, \\ldots, t_k])(y) = f^{(k)}(y)(F(t_1)(y), F(t_2)(y), \\ldots, F(t_m)(y)).

    :param tree: Unlabelled tree corresponding to the elementary differential
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
    if not isinstance(tree, Tree):
        raise TypeError("The argument 'tree' must be of type Tree, not " + str(type(tree)))
    if tree.colors() > 1:
        raise ValueError("Tree passed to elementary differential must be unlabelled.")

    _check_f_y(f, y)

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

            print(bs.series()) # Print the B-Series as a sympy expression
            print(bs(1, 0.1)) # Evaluate the B-Series at y = 1, h = 0.1
    """

    def __init__(self, y : sp.Matrix, f : sp.Matrix, weights : Map, order : int):
        if not isinstance(weights, Map):
            raise TypeError("weights must be a Map, not " + str(type(weights)))
        if not isinstance(order, int):
            raise TypeError("order must be an int, not " + str(type(order)))
        if order < 0:
            raise ValueError("order cannot be negative")

        _check_f_y(f, y)

        self.f = f
        self.y = y
        self.f_imm = sp.ImmutableDenseMatrix(f) #Immutable for cache in elementary_differential
        self.y_imm = sp.ImmutableDenseMatrix(y) #Immutable for cache in elementary_differential
        self.h = sp.symbols('h')
        self.order = order
        self.weights = weights
        self.dim = len(y)
        self.symbolic_expr = sp.zeros(*sp.shape(y))
        for t in trees_up_to_order(order):
            self.symbolic_expr = self.symbolic_expr + self.h ** t.nodes() * weights(t) * _elementary_differential(t, self.f_imm, self.y_imm) / t.sigma()

    def __call__(self, y : list, h : Union[int, float]) -> list:
        """
        Evalutes the B-series at the given values for y and h.

        :param y: List of values to substitute for y
        :type y: list
        :param h: Value to substitute for the step size h
        :type h: int | float
        :return: Value of the B-series evaluated for the given values of y and h
        :rtype: list

        """
        if not isinstance(y, list):
            raise ValueError("y must be a list, not " + str(type(y)))
        if len(y) != self.dim:
            raise ValueError("List of values for y is of incorrect length. Expected " + str(self.dim) + " got " + str(len(y)))
        if not isinstance(h, (int, float)):
            raise ValueError("h must be an int or float, not " + str(type(h)))

        out = self.symbolic_expr.subs(self.h, h)
        for i in range(self.dim):
            out = out.subs(self.y[i], y[i])
        return [float(x) for x in out]

    def series(self) -> sp.Matrix:
        """
        Returns the truncated B-series as a sympy Matrix.

        :rtype: sympy.Matrix
        """
        return self.symbolic_expr

    def __repr__(self):
        return repr(self.symbolic_expr)

    # def __and__(self, other : 'BSeries') -> 'BSeries':
    #     """
    #     Returns the composition of two B-Series, assuming they are with respect
    #     to the same variables and vector field. That is, given two B-Series:
    #
    #     .. math::
    #
    #         B_h(\\varphi, y_0) := \\sum_{|t| \\leq n} \\frac{h^{|t|}}{\\sigma(t)} \\varphi(t) F(t)(y_0),
    #
    #     .. math::
    #
    #         B_h(\\psi, y_0) := \\sum_{|t| \\leq n} \\frac{h^{|t|}}{\\sigma(t)} \\psi(t) F(t)(y_0),
    #
    #     returns their composition :math:`B_h(\\psi, B_h(\\varphi, y_0)) = B_h(\\varphi * \\psi, y_0)`,
    #     where the product is the BCK product of characters. The truncation order of the resulting B-series
    #     is the minimum of the truncation orders of the original two.
    #
    #     :param other: other
    #     :type other: BSeries
    #     """
    #     if not isinstance(other, BSeries):
    #         raise TypeError("Cannot multiply BSeries and object of type " + str(type(other)))
    #     if self.y != other.y:
    #         raise ValueError("Cannot compose B-Series: symbolic variables y of the two series do not match")
    #     if self.f != other.f:
    #         raise ValueError("Cannot compose B-Series: vector fields of the two series do not match")
    #
    #     return BSeries(self.y, self.f, self.weights * other.weights, min(self.order, other.order))
