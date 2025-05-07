"""
Runge-Kutta Schemes
"""
import copy
from typing import Union, Callable, Tuple
import warnings

import numpy as np
import sympy
from scipy.optimize import root
import matplotlib.pyplot as plt
from tqdm import tqdm

from .gentrees import trees_of_order
from .trees import Tree, Forest, ForestSum
from .maps import Map
from .generic_algebra import _apply

def _internal_symbolic(i, t_rep, a, b, s):
    return sum(a[i,j] * _derivative_symbolic(j, t_rep, a, b, s) for j in range(s))

def _derivative_symbolic(i, t_rep, a, b, s):
    if t_rep in (None, []):
        return 1
    out = 1
    for subtree in t_rep[:-1]:
        out *= _internal_symbolic(i, subtree, a, b, s)
    return out

def _elementary_symbolic(t_rep, a, b, s):
    if t_rep is None:
        return 1
    if len(t_rep) == 1:
        return sum(b)
    return sum(b[i] * _derivative_symbolic(i, t_rep, a, b, s) for i in range(s))

def _rk_symbolic_weight(t, s, explicit = False, a_mask = None, b_mask = None):
    if a_mask is None:
        a_mask = [[1 for _ in range(s)] for _ in range(s)]
    if b_mask is None:
        b_mask = [1 for _ in range(s)]
    if explicit:
        for i in range(s):
            for j in range(i, s):
                a_mask[i][j] = 0

    a = sympy.Matrix(s, s, lambda i, j: sympy.symbols(f'a{i}{j}'))
    b = sympy.Matrix(1, s, lambda i, j: sympy.symbols(f'b{j}'))

    for i in range(s):
        for j in range(s):
            if not a_mask[i][j]:
                a[i,j] = 0

    for i in range(s):
        if not b_mask[i]:
            b[i] = 0

    return _elementary_symbolic(t.list_repr, a, b, s)

def rk_symbolic_weight(
        t : Union[Tree, Forest, ForestSum],
        s : int,
        explicit : bool = False,
        a_mask : list = None,
        b_mask : list = None,
        mathematica_code : bool = False,
        rationalise : bool = True
) -> Union[sympy.core.add.Add, str]:
    """
    Returns the elementary weight of a Tree, Forest or ForestSum :math:`t` as a SymPy symbolic expression.

    :param t: A Tree, Forest or ForestSum
    :param s: The number of Runge--Kutta stages
    :type s: int
    :param explicit: If true, assumes the Runge--Kutta scheme is explicit, i.e. :math:`a_{ij} = 0` for :math:`i \\leq j`.
    :type explicit: bool
    :param a_mask: A two-dimensional array specifying which coefficients of the Runge--Kutta parameter matrix :math:`A`
        are non-zero. If not None, sets :math:`a_{ij} = 0` for all :math:`i,j` such that ``A_mask[i][j] = 0``.
    :param b_mask: A one-dimensional array or list specifying which coefficients of the Runge--Kutta parameter vector :math:`b`
        are non-zero. If not None, sets :math:`b_i = 0` for all :math:`i` such that ``b_mask[i] = 0``.
    :param mathematica_code: If true, outputs the expression as mathematica code.
    :type mathematica_code: bool
    :param rationalise: If true, will attempt to rationalise the coefficients in the expression
    :type rationalise: bool
    :returns: The elementary weight of :math:`t`, as a SymPy symbolic expression if `mathematica_code` is False or as a
        string if `mathematica_code` is True.
    :rtype: sympy.core.add.Add | string

    Example usage::

            t = Tree([[],[]])
            RK_symbolic_weight(t, 2) # Returns b0*(a00 + a01)**2 + b1*(a10 + a11)**2
            RK_symbolic_weight(t, 2, explicit = True) # Returns a10**2*b1

            A_mask = [[1,0],[0,1]]
            b_mask = [0,1]
            RK_symbolic_weight(t, 2, A_mask = A_mask, b_mask = b_mask) #Returns a11**2*b1

    .. code-block:: python

        #Generate order conditions as mathematica equations and write to text file

        order_conditions = [Tree([]) - 1.,
                            Tree([[]]) - 1./2,
                            Tree([[],[]]) - 1./3]

        strs = []

        for i,t in enumerate(order_conditions):
            cond = RK_symbolic_weight(t, 3, explicit = True, mathematica_code = True, rationalise = True)
            str_ = "eq" + str(i) + " = " + cond + " == 0; \\n"
            strs.append(str_)

        with open("mathematica_code.txt", "w") as text_file:
            for s in strs:
                text_file.write(s)

    """
    t_ = t
    if isinstance(t, (int, float)):
        t_ = t * Tree(None).as_forest_sum()

    out = _apply(t_, lambda x : _rk_symbolic_weight(x, s, explicit, a_mask, b_mask))

    if rationalise:
        out = sympy.nsimplify(out, tolerance=1e-10, rational = True)

    if mathematica_code:
        out = sympy.mathematica_code(out)
    return out


def rk_order_cond(
        t : Union[Tree, Forest, ForestSum],
        s : int,
        explicit : bool = False,
        a_mask : list = None,
        b_mask : list = None,
        mathematica_code : bool = False,
        rationalise : bool = True
) -> Union[sympy.core.add.Add, str]:
    """
    Returns the Runge--Kutta order condition associated with tree :math:`t` as a SymPy symbolic expression.

    :param t: A Tree
    :param s: The number of Runge--Kutta stages
    :type s: int
    :param explicit: If true, assumes the Runge--Kutta scheme is explicit, i.e. :math:`a_{ij} = 0` for :math:`i \\leq j`.
    :type explicit: bool
    :param a_mask: A two-dimensional array specifying which coefficients of the Runge--Kutta parameter matrix :math:`A`
        are non-zero. If not None, sets :math:`a_{ij} = 0` for all :math:`i,j` such that ``A_mask[i][j] = 0``.
    :param b_mask: A one-dimensional array or list specifying which coefficients of the Runge--Kutta parameter vector :math:`b`
        are non-zero. If not None, sets :math:`b_i = 0` for all :math:`i` such that ``b_mask[i] = 0``.
    :param mathematica_code: If true, outputs the expression as mathematica code.
    :type mathematica_code: bool
    :param rationalise: If true, will attempt to rationalise the coefficients in the expression
    :type rationalise: bool
    :returns: The order condition associated with the tree :math:`t`, as a SymPy symbolic expression if `mathematica_code` is False or as a
        string if `mathematica_code` is True.
    :rtype: sympy.core.add.Add | string

    Example usage::

            t = Tree([[],[]])
            RK_order_cond(t, 2) # Returns b0*(a00 + a01)**2 + b1*(a10 + a11)**2 - 1/3
            RK_order_cond(t, 2, explicit = True) # Returns a10**2*b1 - 1/3

            A_mask = [[1,0],[0,1]]
            b_mask = [0,1]
            RK_order_cond(t, 2, A_mask = A_mask, b_mask = b_mask) #Returns a11**2*b1 - 1/3

    .. code-block:: python

        #Generate order conditions as mathematica equations and write to text file

        strs = []

        for i,t in enumerate(trees_of_order(4)):
            cond = RK_symbolic_weight(t, 3, explicit = True, mathematica_code = True, rationalise = True)
            str_ = "eq" + str(i) + " = " + cond + " == 0; \\n"
            strs.append(str_)

        with open("mathematica_code.txt", "w") as text_file:
            for s in strs:
                text_file.write(s)

    """
    return rk_symbolic_weight(t - 1. / t.factorial(), s, explicit, a_mask, b_mask, mathematica_code, rationalise)

class RK:
    """
    A Runge--Kutta method with the Butcher tableau:

    .. math::

        \\begin{array}{c|c}
            c & A \\\\
            \\hline
             & b^T
        \\end{array}

    where :math:`c_i = \\sum_{j=1}^s a_{ij}`.

    :param a: The Runge--Kutta parameter matrix :math:`A`.
    :param b: The Runge--Kutta parameter vector :math:`b`.
    """
    def __init__(self, a, b, name = None):
        self.name = name
        self.s = len(b)
        if len(a) != self.s or len(a[0]) != self.s:
            raise ValueError("Parameter 'a' must be a square s x s matrix and b a vector of length s")

        self.a = a
        self.b = b
        self.c = [sum(a[i][j] for j in range(self.s)) for i in range(self.s)]

        self.explicit = self._check_explicit()
        self.deriv_dict = {}  # {repr(None) : 1, repr([]) : 1}
        for i in range(self.s):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

        self.np_a = np.array(a)
        self.np_b = np.array(b)

    def __repr__(self):
        out = "["
        for i in range(self.s - 1):
            out += repr(self.a[i]) + ",\n"
        out += repr(self.a[-1]) + "]\n"
        out += repr(self.b)
        return out

    def _check_explicit(self):
        for i in range(self.s):
            for j in range(i, self.s):
                if self.a[i][j]:
                    return False
        return True

    def _inverse(self):
        b_inv = [-self.b[i] for i in range(self.s)]
        a_inv = [[self.a[i][j] - self.b[j] for j in range(self.s)] for i in range(self.s)]
        return RK(a_inv, b_inv)

    def reverse(self) -> 'RK':
        """
        Returns the RK scheme given by reversing the step size h to -h, with Butcher tableau:

        .. math::

            \\begin{array}{c|c}
                -c & -A \\\\
                \\hline
                 & -b^T
            \\end{array}

        :rtype: RK
        """
        return RK([[-self.a[i][j] for j in range(self.s)] for i in range(self.s)], [-self.b[i] for i in range(self.s)])

    def adjoint(self) -> 'RK':
        """
        Returns the adjoint Runge--Kutta method, given by the Butcher tableau:

        .. math::

            \\begin{array}{c|c}
                \\widetilde{c} & e \\widetilde{b}^T - \\widetilde{A} \\\\
                \\hline
                 & \\widetilde{b}^T
            \\end{array}

        where :math:`\\widetilde{b}_i := b_{s+1-i}` and :math:`\\widetilde{A}_{ij} := A_{s+1 - i, s+ 1 - j}` for all
        :math:`1 \\leq i,j \\leq s`.

        :rtype: RK
        """
        b_adj = [self.b[self.s - 1 - j] for j in range(self.s)]
        a_adj = [[self.b[self.s - 1 - j] - self.a[self.s - 1 - i][self.s - j - 1] for j in range(self.s)] for i in range(self.s)]
        return RK(a_adj, b_adj)

    def _explicit_step(self, y0, t0, f, h):
        k = [None] * self.s

        for i in range(self.s):
            y_stage = y0 + h * sum(self.a[i][j] * k[j] for j in range(i))
            k[i] = f(t0 + self.c[i] * h, y_stage)

        y_next = y0 + h * sum(self.b[i] * k[i] for i in range(self.s))
        return y_next

    # def _explicit_step(self, y0, t0, f, h):
    #     k = np.empty(self.s)
    #
    #     for i in range(self.s):
    #         y_stage = y0 + h * np.sum(self.np_a @ k)
    #         k[i] = f(t0 + self.c[i] * h, y_stage)
    #
    #     y_next = y0 + h * np.dot(self.np_b, k)
    #     return y_next

    def _implicit_step(self, y0, t0, f, h, tol = 1e-10, max_iter = 100):
        y0 = np.array(y0)
        dim = len(y0)

        # Start with all stages equal f(t_n, y_n)
        k0 = np.tile(f(t0, y0), self.s)

        def G(K_flat):
            K = K_flat.reshape((self.s, dim))
            G_vec = []

            for i in range(self.s):
                y_stage = y0 + h * sum(self.a[i][j] * K[j] for j in range(self.s))
                t_stage = t0 + self.c[i] * h
                G_i = K[i] - f(t_stage, y_stage)
                G_vec.append(G_i)

            return np.concatenate(G_vec)

        sol = root(G, k0, method='hybr', tol=tol, options={'maxfev': max_iter})

        if not sol.success:
            warnings.warn(f"Implicit RK solver failed: {sol.message}")

        K = sol.x.reshape((self.s, dim))
        y_next = y0 + h * sum(self.b[i] * K[i] for i in range(self.s))
        return y_next

    def step(self,
             y0 : Union[list, np.array],
             t0 : float,
             f : Callable[[float, float], Union[list, np.array]],
             h : float,
             tol : float = 1e-10,
             max_iter : int = 100
             ) -> Union[list, np.array]:
        """
        Runs one step of the Runge--Kutta method.

        :param y0: Initial condition for y
        :type y0: list | array
        :param t0: Initial condition for t
        :type t0: float
        :param f: Function defining the ODE :math:`dy / dt = f(t,y)`.
        :type f: callable
        :param h: Step size
        :type h: float
        :param tol: Tolerance for the root solving algorithm. Only applicable if the scheme is implicit.
        :type tol: float
        :param max_iter: Maximum number of iterations for the root solving algorithm. Only applicable if the scheme is implicit.
        :type max_iter: int
        :return: Next point, y1
        :rtype: list | array
        """
        def f_(t_, y_):
            return np.array(f(t_,y_))
        y0_ = np.array(y0).copy()

        if self.explicit:
            return self._explicit_step(y0_, t0, f_, h)

        return self._implicit_step(y0_, t0, f_, h, tol, max_iter)

    def run(self,
            y0 : Union[list, np.array],
            t0 : float,
            t_end : float,
            f : Callable[[float, float], Union[list, np.array]],
            n : int,
            tol : float = 1e-10,
            max_iter : int = 100,
            plot : bool = False,
            plot_dims : bool = None,
            plot_kwargs : dict = None
            ) -> Tuple[list, list]:
        """
        Runs the Runge--Kutta method.

        :param y0: Initial condition for y
        :type y0: list | array
        :param t0: Initial condition for t
        :type t0: float
        :param t_end: End point for t
        :type t_end: float
        :param f: Function defining the ODE :math:`dy / dt = f(t,y)`.
        :type f: callable
        :param n: Number of steps
        :type n: int
        :param tol: Tolerance for the root solving algorithm. Only applicable if the scheme is implicit.
        :type tol: float
        :param max_iter: Maximum number of iterations for the root solving algorithm. Only applicable if the scheme is implicit.
        :type max_iter: int
        :param plot: If true, will plot the solution
        :type plot: bool
        :param plot_dims: List of dimensions of the solution to plot
        :type plot_dims: list | array
        :param plot_kwargs: kwargs to pass to pyplot.plot() if plotting the solution.
        :type plot_kwargs: dict

        :return: t_vals, y_vals - the lists of values of t and y respectively
        :rtype: tuple[list, list]
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if plot_dims is None:
            plot_dims = list(range(len(y0)))

        def f_(t_, y_):
            return np.array(f(t_, y_))
        y0_ = np.array(y0).copy()

        t_vals = [t0]
        y_vals = [y0_]

        t = t0
        y = y0_.copy()
        h = (t_end - t0) / n

        step_func = (lambda y_, t_ : self._explicit_step(y_, t_, f_, h)) if self.explicit else (lambda y_, t_ : self._implicit_step(y_, t_, f_, h, tol, max_iter))

        for _ in tqdm(range(n)):
            y = step_func(y, t)
            t += h
            t_vals.append(t)
            y_vals.append(copy.deepcopy(y))

        if plot:
            plt.plot(t_vals, np.array(y_vals)[:, plot_dims], **plot_kwargs)

        return t_vals, y_vals

    def __add__(self, other : 'RK') -> 'RK':
        """
        Returns the sum of two RK schemes, :math:`(A_1, b_1)` and :math:`(A_2, b_2)`, with Butcher tableau:

        .. math::

            \\begin{array}{c|cc}
                c_1 & A_1 & 0 \\\\
                c_2 & 0 & A_2\\\\
                \\hline
                 & b_1 & b_2
            \\end{array}

        :rtype: RK
        """
        s1 = other.s
        a1 = other.a
        b1 = other.b

        s2 = self.s
        a2 = self.a
        b2 = self.b

        a = [[a1[i][j] for j in range(s1)] + [0 for j in range(s2)] for i in range(s1)]
        a += [[0 for j in range(s1)] + [a2[i][j] for j in range(s2)] for i in range(s2)]
        b = b1 + b2

        return RK(a, b)

    def __neg__(self):
        return RK(self.a, [-self.b[i] for i in range(self.s)])

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, other : 'RK') -> 'RK':
        """
        Returns the composition of two RK schemes, :math:`(A_1, b_1)` and :math:`(A_2, b_2)`, with Butcher tableau:

        .. math::

            \\begin{array}{c|cc}
                c_1 & A_1 & 0 \\\\
                c_2 & e b_1^T & A_2\\\\
                \\hline
                 & b_1 & b_2
            \\end{array}

        where :math:`e` is the vector of 1s.

        :rtype: RK
        """
        s1 = other.s
        a1 = other.a
        b1 = other.b

        s2 = self.s
        a2 = self.a
        b2 = self.b

        a = [[a1[i][j] for j in range(s1)] + [0 for j in range(s2)] for i in range(s1)]
        a += [[b1[j] for j in range(s1)] + [a2[i][j] for j in range(s2)] for i in range(s2)]
        b = b1 + b2

        return RK(a,b)

    def __pow__(self, exponent : int) -> 'RK':
        """
        Returns the compositional power of the Runge--Kutta scheme. In particular, ``self ** (-1)`` returns the scheme
        with Butcher tableau:

        .. math::

            \\begin{array}{c|c}
                 & A - e b^T \\\\
                \\hline
                 & -b^T
            \\end{array}

        where :math:`e` is the vector of 1s.

        :param exponent: Exponent
        :type exponent: int
        :rtype: RK
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent in RK power must be int, got " + str(type(exponent)) + " instead.")

        if exponent == 0:
            return RK([[0]], [0])

        expn_ = exponent
        if exponent < 0:
            out = self._inverse()
            expn_ = -exponent
        else:
            out = copy.deepcopy(self)

        for _ in range(expn_-1):
            out = out * self
        return out

    def _internal_weights(self, i, t_rep):
        return sum(self.a[i][j] * self._derivative_weights(j, t_rep) for j in range(self.s))

    def _derivative_weights(self, i, t_rep):
        if (i, repr(t_rep)) in self.deriv_dict:
            return self.deriv_dict[(i, repr(t_rep))]

        out = 1
        for subtree in t_rep[:-1]:
            out *= self._internal_weights(i, subtree)
        self.deriv_dict[(i, repr(t_rep))] = out
        return out

    def _elementary_weights(self, t_rep):
        if t_rep is None:
            return 1
        return sum(self.b[i] * self._derivative_weights(i, t_rep) for i in range(self.s))

    def elementary_weights_map(self) -> Map:
        """
        Returns the elementary weight function of the Runge-Kutta method as an instance of the Map class.

        :rtype: Map
        """
        def f_(x):
            return self._elementary_weights(x.list_repr)
        return Map(f_)

    def modified_equation_map(self) -> Map:
        """
        Returns the map corresponding to the elementary weights function of the
        modified (B-series) vector field, :math:`\\widetilde{\\phi}`, defined by

        .. math::

            (\\widetilde{\\phi} \\star e)(t) = \\phi(t)

        where :math:`\\phi` is the elementary weights function of the Runge-Kutta
        scheme and :math:`e(t) = 1 / t!` is the elementary weights function of
        the exact solution. Equivalently,

        .. math::

            \\widetilde{\\phi}(t) = (\\phi \\star e^{\\star (-1)})(t).

        :return: Elementary weights map of the modified vector field
        :rtype: Map
        """
        return self.elementary_weights_map().modified_equation()

    def order(self, tol : float = 1e-15) -> int:
        """
        Returns the order of the RK scheme.

        :param tol: Tolerance for evaluating order conditions. An order condition of the form ``self.elementary_weights(t) = 1./t.factorial()``
            is considered to be satisfied if ``abs( self.elementary_weights(t) - 1./t.factorial() ) > tol``
        :type tol: float
        :rtype: int
        """
        n = 0
        while True:
            for t in trees_of_order(n):
                if abs(self._elementary_weights(t.list_repr) - 1. / t.factorial()) > tol:
                    return n-1
            n += 1
