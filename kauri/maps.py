from .trees import Tree, Forest, ForestSum
from .gentrees import trees_of_order
import copy

class Map:
    """
    A multiplicative linear map over the Hopf algebra of planar rooted trees. This class is callable, allowing it to be
    used in conjunction with the ``.apply()``, ``.apply_product()`` and ``.apply_power()`` methods of the classes ``Tree``,
    ``Forest`` and ``ForestSum``.

    :param func: A function taking as input a single tree and returning a scalar, Tree, Forest or ForestSum.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, t):
        return t.apply(self.func)

    def __pow__(self, n):
        """
        Returns the power of the map, where the product of functions is defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        and negative powers are defined as :math:`f^{-n} = f^n \\circ S`, where :math:`S` is the antipode.

        :param n: Exponent
        :type n: int
        :rtype: Map

        .. note::
            ``f ** n`` will call ``.apply_power()`` methods with ``apply_reduction = True``. If speed is critical,
            consider defining the power manually as ``Map(lambda x : x.apply_power(f, n, apply_reduction = False))``
            and calling ``.reduce()`` on the final result of the computation.

        Example usage::

            ident = Map(lambda x : x)
            S = ident ** (-1) # antipode
            ident_sq = ident ** 2 # identity squared
        """
        if not isinstance(n, int):
            raise ValueError("Map.__pow__ received invalid exponent")

        return Map(lambda x : x.apply_power(self.func, n))

    def __imul__(self, other):
        func_ = self.func
        if isinstance(other, int) or isinstance(other, float):
            self.func = lambda x : other * func_(x)
        elif isinstance(other, Map):
            self.func = lambda x: x.apply_product(func_, other.func)
        else:
            raise
        return self

    def __ixor__(self, other):
        func_ = self.func
        if isinstance(other, Map):
            self.func = lambda x: x.apply_cem_product(func_, other.func)
        else:
            raise
        return self

    def __mul__(self, other):
        """
        Returns the product of maps, defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta (t)

        :type other: Map
        :rtype: Map

        .. note::
            ``f * g`` will call ``.apply_product()`` methods with ``apply_reduction = True``. If speed is critical,
            consider defining the power manually as ``Map(lambda x : x.apply_product(f, g, apply_reduction = False))``
            and calling ``.reduce()`` on the final result of the computation.

        Example usage::

            ident = Map(lambda x : x)
            S = Map(lambda x : x.antipode())
            counit = ident * S
        """
        temp = copy.deepcopy(self)
        temp *= other
        return temp

    def __xor__(self, other):
        temp = copy.deepcopy(self)
        temp ^= other
        return temp

    def __iadd__(self, other):
        func_ = self.func
        if isinstance(other, Map):
            self.func = lambda x: func_(x) + other.func(x)
        else:
            raise
        return self

    def __add__(self, other):
        """
        Returns the pointwise sum of two maps, given by

        .. math::

            (f + g)(t) := f(t) + g(t)

        :type other: Map
        :rtype: Map
        """
        temp = copy.deepcopy(self)
        temp += other
        return temp

    def __neg__(self):
        return Map(lambda x : -self.func(x))

    def __isub__(self, other):
        self.__iadd__(-other)
        return self

    def __sub__(self, other):
        temp = copy.deepcopy(self)
        temp -= other
        return temp

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__

    def __matmul__(self, other):
        """
        Returns the composition of two maps, given by

        .. math::

            (f \\, @ \\, g)(t) := (f \\circ g)(t) := f(g(t))

        :type other: Map
        :rtype: Map

        .. note::
            ``f @ g`` will call ``.apply()`` methods with ``apply_reduction = True``. If speed is critical,
            consider defining the composition manually as ``Map(lambda x : self(x).apply(other, apply_reduction = False))``
            and calling ``.reduce()`` on the final result of the computation.
        """
        return Map(lambda x : (other(x) * Tree(None)).apply(self))

    def modified_equation(self):
        """
        Assuming the given map :math:`\\phi` corresponds to the elementary weights function of a B-series method, returns the map
        corresponding to the elementary weights function of the modified (B-series) vector field, :math:`\\widetilde{\\phi}`,
        defined by

        .. math::

            (\\widetilde{\\phi} \\star e)(t) = \\phi(t)

        where :math:`e(t) = 1 / t!` is the elementary weights function of the exact solution, or equivalently

        .. math::

            \\widetilde{\\phi}(t) = (\\phi \\star e^{\\star (-1)})(t)

        where :math:`\\mathrm{Id}` is the identity map on trees and :math:`e^{\\star (-1)} = e \\circ S_{CEM}` :cite:`chartier2010algebraic`.

        :param apply_reduction: If set to True (default), will simplify the output by cancelling terms where applicable.
            Should be set to False if being used as part of a larger computation, to avoid time-consuming premature simplifications.
        :return: Elementary weights map of the modified vector field
        """
        return self.logarithm()

    def preprocessed_integrator(self, apply_reduction = True):
        #TODO
        return exact_weights ^ (self @ S_CEM)

    def exponential(self):
        """
        Returns the exponential of the map, defined as

        .. math::

            \\exp(\\phi) = \\phi \\star e

        where :math:`e(t) = 1 / t!` is the elementary weights function of the exact solution :cite:`chartier2010algebraic, murua2006hopf`.

        :return: Exponential map
        :rtype: Map
        """
        return self ^ exact_weights

    def logarithm(self):
        """
        Returns the logarithm of the map, defined as

        .. math::

            \\log(\\phi) = \\phi \\star e^{\\star (-1)}

        where :math:`e(t) = 1 / t!` is the elementary weights function of the exact solution and :math:`e^{\\star (-1)} = e \\circ S_{CEM}`
        :cite:`chartier2010algebraic, murua2006hopf`.

        :return: Exponential map
        :rtype: Map
        """
        return self ^ (exact_weights @ S_CEM)


# Some common examples provided for convenience
ident = Map(lambda x : x)
counit = Map(lambda x : 1 if x == Tree(None) else 0)
counit_CEM = Map(lambda x : 1 if x == Tree([]) else 0)
S = Map(lambda x : x.antipode())
S_CEM = Map(lambda x : x.cem_antipode())
exact_weights = Map(lambda x : 1. / x.factorial())

omega = Map(lambda x : 1 if x == Tree(None) or x == Tree([]) else 0).logarithm()