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

def RK_symbolic_weight(t, s, explicit = False, A_mask = None, b_mask = None):
    """
    Returns the elementary weight as a symbolic expression

    :param t: A forest sum
    :param s:
    :param explicit:
    :param A_mask:
    :param b_mask:
    :return:
    """

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

def RK_order_cond(t, s, explicit = False, A_mask = None, b_mask = None):
    return RK_symbolic_weight(t, s, explicit, A_mask, b_mask) - 1./t.factorial()

class RK:
    def __init__(self, A, b):
        self.s = len(b)
        if len(A) != self.s or len(A[0]) != self.s:
            raise ValueError("A must be a square s x s matrix and b a vector of length s")

        self.A = A
        self.b = b
        self.c = [sum(A[i][j] for j in range(self.s)) for i in range(self.s)]

        self.np_A = np.array(self.A)
        self.np_b = np.array(self.b)[:, np.newaxis]
        self.np_c = np.array(self.c)

        self.explicit = self.check_explicit()
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

    def check_explicit(self):
        for i in range(self.s):
            for j in range(i, self.s):
                if self.A[i][j]:
                    return False
        return True

    def reduce(self):
        """
        P-reduces the method in-place
        :return:
        """
        pass

    def inverse(self):
        b_inv = [-self.b[i] for i in range(self.s)]
        A_inv = [[self.A[i][j] - self.b[j] for j in range(self.s)] for i in range(self.s)]
        return RK(A_inv, b_inv)

    def reverse(self):
        """
        Returns the RK scheme given by reversing the step size h to -h.
        :return:
        """
        return RK([[-self.A[i][j] for j in range(self.s)] for i in range(self.s)], [-self.b[i] for i in range(self.s)])

    def adjoint(self):
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
        f_ = lambda t_,y_ : np.array(f(t_,y_))
        y0_ = np.array(y0).copy()

        if self.explicit:
            return self._explicit_step(y0_, t0, f_, h)
        else:
            return self._implicit_step(y0_, t0, f_, h, tol, max_iter)

    def run(self, y0, t0, t_end, f, n, tol = 1e-10, max_iter = 100, plot = False, plot_dims = None, plot_kwargs=None):
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
        Addition
        :return:
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
        Composition
        :return:
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
        pass

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
        return self._elementary_weights(t.list_repr)

    def order(self, tol = 1e-15):
        n = 0
        while True:
            for t in trees_of_order(n):
                if abs(self.elementary_weights(t) - 1. / t.factorial()) > tol:
                    return n-1
            n += 1