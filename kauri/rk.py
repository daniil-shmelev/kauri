from .gentrees import trees_of_order
from .trees import Tree
import numpy as np
import sympy as sp
from scipy.optimize import root, fsolve
import copy
import matplotlib.pyplot as plt

def _internal_symbolic(i, t_rep, A, b, s):
    return sum(A[i,j] * _derivative_symbolic(j, t_rep, A, b, s) for j in range(s))

def _derivative_symbolic(i, t_rep, A, b, s):
    if t_rep == None or t_rep == []:
        return 1
    out = 1
    for subtree in t_rep:
        out *= _internal_symbolic(i, subtree, A, b, s)
    return out

def _elementary_symbolic(t_rep, A, b, s):
    if t_rep is None:
        return 1
    if t_rep == []:
        return sum(b)
    return sum(b[i] * _derivative_symbolic(i, t_rep, A, b, s) for i in range(s))

def _RK_symbolic_weight(t, s, explicit = False, A_mask = None, b_mask = None):
    if A_mask is None:
        A_mask = [[1 for j in range(s)] for i in range(s)]
    if b_mask is None:
        b_mask = [1 for i in range(s)]
    if explicit:
        for i in range(s):
            for j in range(i, s):
                A_mask[i][j] = 0

    A = sp.Matrix(s, s, lambda i, j: sp.symbols(f'a{i}{j}'))
    b = sp.Matrix(1, s, lambda i, j: sp.symbols(f'b{j}'))

    for i in range(s):
        for j in range(s):
            if not A_mask[i][j]:
                A[i,j] = 0

    for i in range(s):
        if not b_mask[i]:
            b[i] = 0

    return _elementary_symbolic(t.list_repr, A, b, s)

def RK_symbolic_weight(t, s, explicit = False, A_mask = None, b_mask = None, mathematica_code = False, rationalise = True):
    """
    Returns the elementary weight of a Tree, Forest or ForestSum :math:`t` as a SymPy symbolic expression.

    :param t: A Tree, Forest or ForestSum
    :param s: The number of Runge--Kutta stages
    :type s: int
    :param explicit: If true, assumes the Runge--Kutta scheme is explicit, i.e. :math:`a_{ij} = 0` for :math:`i \\leq j`.
    :type explicit: bool
    :param A_mask: A two-dimensional array specifying which coefficients of the Runge--Kutta parameter matrix :math:`A`
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
    if isinstance(t, int) or isinstance(t, float):
        t_ = t * Tree(None).as_forest_sum()

    out = t_.apply(lambda x : _RK_symbolic_weight(x, s, explicit, A_mask, b_mask), apply_reduction = False)

    if rationalise:
        out = sp.nsimplify(out, tolerance=1e-10, rational = True)

    if mathematica_code:
        out = sp.mathematica_code(out)
    return out


def RK_order_cond(t, s, explicit=False, A_mask=None, b_mask=None, mathematica_code = False, rationalise = True):
    """
    Returns the Runge--Kutta order condition associated with tree :math:`t` as a SymPy symbolic expression.

    :param t: A Tree
    :param s: The number of Runge--Kutta stages
    :type s: int
    :param explicit: If true, assumes the Runge--Kutta scheme is explicit, i.e. :math:`a_{ij} = 0` for :math:`i \\leq j`.
    :type explicit: bool
    :param A_mask: A two-dimensional array specifying which coefficients of the Runge--Kutta parameter matrix :math:`A`
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
    return RK_symbolic_weight(t - 1. / t.factorial(), s, explicit, A_mask, b_mask, mathematica_code, rationalise)

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

    :param A: The Runge--Kutta parameter matrix :math:`A`.
    :param b: The Runge--Kutta parameter vector :math:`b`.
    """
    def __init__(self, A, b):
        self.s = len(b)
        if len(A) != self.s or len(A[0]) != self.s:
            raise ValueError("A must be a square s x s matrix and b a vector of length s")

        self.A = A
        self.b = b
        self.c = [sum(A[i][j] for j in range(self.s)) for i in range(self.s)]

        self.explicit = self._check_explicit()
        self.deriv_dict = {}  # {repr(None) : 1, repr([]) : 1}
        for i in range(self.s):
            self.deriv_dict[(i, repr(None))] = 1
            self.deriv_dict[(i, repr([]))] = 1

    def __repr__(self):
        out = "["
        for i in range(self.s - 1):
            out += repr(self.A[i]) + ",\n"
        out += repr(self.A[-1]) + "]\n"
        out += repr(self.b)
        return out

    def _check_explicit(self):
        for i in range(self.s):
            for j in range(i, self.s):
                if self.A[i][j]:
                    return False
        return True

    def _inverse(self):
        b_inv = [-self.b[i] for i in range(self.s)]
        A_inv = [[self.A[i][j] - self.b[j] for j in range(self.s)] for i in range(self.s)]
        return RK(A_inv, b_inv)

    def reverse(self):
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
        return RK([[-self.A[i][j] for j in range(self.s)] for i in range(self.s)], [-self.b[i] for i in range(self.s)])

    def adjoint(self):
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
        A_adj = [[self.b[self.s - 1 - j] - self.A[self.s - 1 - i][self.s - j - 1] for j in range(self.s)] for i in range(self.s)]
        return RK(A_adj, b_adj)

    def _explicit_step(self, y0, t0, f, h):
        k = [None] * self.s

        for i in range(self.s):
            y_stage = y0 + h * sum(self.A[i][j] * k[j] for j in range(i))
            k[i] = f(t0 + self.c[i] * h, y_stage)

        y_next = y0 + h * sum(self.b[i] * k[i] for i in range(self.s))
        return y_next

    def _implicit_step(self, y0, t0, f, h, tol = 1e-10, max_iter = 100):
        y0 = np.array(y0)
        dim = len(y0)

        # Start with all stages equal f(t_n, y_n)
        k0 = np.tile(f(t0, y0), self.s)

        def G(K_flat):
            K = K_flat.reshape((self.s, dim))
            G_vec = []

            for i in range(self.s):
                y_stage = y0 + h * sum(self.A[i][j] * K[j] for j in range(self.s))
                t_stage = t0 + self.c[i] * h
                G_i = K[i] - f(t_stage, y_stage)
                G_vec.append(G_i)

            return np.concatenate(G_vec)

        sol = root(G, k0, method='hybr', tol=tol, options={'maxfev': max_iter})

        if not sol.success:
            raise RuntimeError(f"Implicit RK solver failed: {sol.message}")

        K = sol.x.reshape((self.s, dim))
        y_next = y0 + h * sum(self.b[i] * K[i] for i in range(self.s))
        return y_next

    def step(self, y0, t0, f, h, tol = 1e-10, max_iter = 100):
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
        f_ = lambda t_,y_ : np.array(f(t_,y_))
        y0_ = np.array(y0).copy()

        if self.explicit:
            return self._explicit_step(y0_, t0, f_, h)
        else:
            return self._implicit_step(y0_, t0, f_, h, tol, max_iter)

    def run(self, y0, t0, t_end, f, n, tol = 1e-10, max_iter = 100, plot = False, plot_dims = None, plot_kwargs=None):
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
            plot_dims = [i for i in range(len(y0))]

        f_ = lambda t_, y_: np.array(f(t_, y_))
        y0_ = np.array(y0).copy()

        t_vals = [t0]
        y_vals = [y0_]

        t = t0
        y = y0_.copy()
        h = (t_end - t0) / n

        step_func = (lambda y_, t_ : self._explicit_step(y_, t_, f_, h)) if self.explicit else (lambda y_, t_ : self._implicit_step(y_, t_, f_, h, tol, max_iter))

        for i in range(n):
            y = step_func(y, t)
            t += h
            t_vals.append(t)
            y_vals.append(copy.deepcopy(y))

        if plot:
            plt.plot(t_vals, np.array(y_vals)[:, plot_dims], **plot_kwargs)

        return t_vals, y_vals

    def __add__(self, other):
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
        A1 = other.A
        b1 = other.b

        s2 = self.s
        A2 = self.A
        b2 = self.b

        A = [[A1[i][j] for j in range(s1)] + [0 for j in range(s2)] for i in range(s1)]
        A += [[0 for j in range(s1)] + [A2[i][j] for j in range(s2)] for i in range(s2)]
        b = b1 + b2

        return RK(A, b)

    def __neg__(self):
        return RK(self.A, [-self.b[i] for i in range(self.s)])

    def __sub__(self, other):
        return self + other.__neg__()

    def __mul__(self, other):
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
        A1 = other.A
        b1 = other.b

        s2 = self.s
        A2 = self.A
        b2 = self.b

        A = [[A1[i][j] for j in range(s1)] + [0 for j in range(s2)] for i in range(s1)]
        A += [[b1[j] for j in range(s1)] + [A2[i][j] for j in range(s2)] for i in range(s2)]
        b = b1 + b2

        return RK(A,b)

    def __pow__(self, n):
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

        :param n: Exponent
        :type n: int
        :rtype: RK
        """
        if not isinstance(n, int):
            raise ValueError("RK.__pow__ received invalid exponent")

        if n == 0:
            return RK([[0]], [0])

        out = None
        n_ = n
        if n < 0:
            out = self._inverse()
            n_ = -n
        else:
            out = copy.deepcopy(self)

        for i in range(n_-1):
            out = out * self
        return out

    def _internal_weights(self, i, t_rep):
        return sum(self.A[i][j] * self._derivative_weights(j, t_rep) for j in range(self.s))

    def _derivative_weights(self, i, t_rep):
        if (i, repr(t_rep)) in self.deriv_dict.keys():
            return self.deriv_dict[(i, repr(t_rep))]
        else:
            out = 1
            for subtree in t_rep:
                out *= self._internal_weights(i, subtree)
            self.deriv_dict[(i, repr(t_rep))] = out
            return out

    def _elementary_weights(self, t_rep):
        if t_rep is None:
            return 1
        return sum(self.b[i] * self._derivative_weights(i, t_rep) for i in range(self.s))

    def elementary_weights(self, t):
        """
        Returns the elementary weight function of the Runge-Kutta method applied to a Tree, Forest or ForestSum :math:`t`.

        :param t: Tree, Forest or ForestSum
        :rtype: float
        """
        return self._elementary_weights(t.list_repr)

    def modified_equation_weights(self, t):
        #TODO
        return t.modified_equation_term().apply(self.elementary_weights())

    def order(self, tol = 1e-15):
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
                if abs(self.elementary_weights(t) - 1. / t.factorial()) > tol:
                    return n-1
            n += 1