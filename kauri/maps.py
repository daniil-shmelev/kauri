"""
This module provides the :class:`Map` class, which implements linear multiplicative maps on trees
and allows for their manipulation with respect to different Hopf algebras. In particular, this covers
characters on the Hopf algebra, as well as more complicated maps.
"""
import copy
from functools import lru_cache
from typing import Union, Callable

from .trees import Tree, Forest, ForestSum, _is_reducible, EMPTY_TREE, _is_tree_like
from .generic_algebra import _apply, _func_power, _func_product

from .bck_impl import _coproduct as bck_coproduct
from .bck_impl import _antipode as bck_antipode
from .bck_impl import _counit as bck_counit

from .cem_impl import _coproduct as cem_coproduct
from .cem_impl import _antipode as cem_antipode

class Map:
    """
    A multiplicative linear map over the Hopf algebra of non-planar rooted trees.
    This class is callable.

    :param func: A function taking as input a single tree and returning a scalar,
        Tree, Forest or ForestSum.
    :type func: Callable[[Tree], int | float | Tree | Forest | ForestSum]
    """
    def __init__(self, func : Callable[[Tree], Union[int, float, Tree, Forest, ForestSum]]):
        #TODO: typecheck?
        self.func = func

    @lru_cache(maxsize = 128) # maxsize here since caching prevents the object being garbage collected
    def __call__(self, t : Union[Tree, Forest, ForestSum]) -> Union[int, float, Tree, Forest, ForestSum]:
        if not _is_tree_like(t):
            raise TypeError("Argument to Map must be Tree, Forest or ForestSum, not " + str(type(t)))
        return _apply(t, self.func)

    def __pow__(self, exponent : int) -> 'Map':
        """
        Returns the power of the map in the BCK Hopf algebra, where the product
        of functions is defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

        and negative powers are defined as :math:`f^{-n} = f^n \\circ S_{BCK}`,
         where :math:`S_{BCK}` is the BCK antipode.

        :param exponent: Exponent
        :type exponent: int
        :rtype: Map

        Example usage::

            import kauri as kr

            ident = kr.Map(lambda x : x)
            S = ident ** (-1) # BCK antipode
            ident_sq = ident ** 2 # identity squared
        """
        if not isinstance(exponent, int):
            raise TypeError("Error in BCK power: exponent must be an integer, got " + str(type(exponent)) + " instead")

        return Map(lambda x : _func_power(x, self.func, exponent, bck_coproduct, bck_counit, bck_antipode))

    def __imul__(self, other : Union[int, float, 'Map']):
        func_ = self.func
        if isinstance(other, (int, float)):
            self.func = lambda x : other * func_(x)
        elif isinstance(other, Map):
            self.func = lambda x : _func_product(x, func_, other.func, bck_coproduct)
        else:
            raise TypeError("Error in BCK product: Cannot multiply Map by object of type " + str(type(other)))
        return self

    def __ixor__(self, other):
        func_ = self.func
        if isinstance(other, (int, float)):
            self.func = lambda x: other * func_(x)
        elif isinstance(other, Map):
            def f_(x):
                if x.list_repr is None:
                    out = other.func(EMPTY_TREE)
                else:
                    out = _func_product(x, func_, other.func, cem_coproduct)
                if _is_reducible(out):
                    out = out.singleton_reduced()
                return out
            self.func = f_
        else:
            raise TypeError("Error in CEM product: Cannot multiply Map by object of type " + str(type(other)))
        return self

    def __mul__(self, other : Union['Map', int, float]) -> 'Map':
        """
        Returns the product of maps in the BCK Hopf algebra, defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t).

        If `other` is of type `int` or `float`, returns the map scaled by `other`.

        :param other: Map | int | float
        :rtype: Map

        Example usage::

            import kauri as kr
            import kauri.bck as bck

            ident = kr.Map(lambda x : x)
            counit = ident * bck.antipode
            ident_2 = 2 * ident # ident_2(t) = 2 * t for any tree t
        """
        temp = copy.deepcopy(self)
        temp *= other
        return temp

    def __xor__(self, other : Union['Map', int, float]) -> 'Map':
        """
        Returns the product of maps in the CEM Hopf algebra, defined by

        .. math::

            (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{CEM} (t).

        If `other` is of type `int` or `float`, returns the map scaled by `other`.

        :param other: Map | int | float
        :rtype: Map

        Example usage::

            import kauri as kr
            import kauri.cem as cem

            ident = kr.Map(lambda x : x)
            counit = ident ^ cem.antipode
            ident_2 = 2 ^ ident # ident_2(t) = 2 * t for any tree t
        """
        temp = copy.deepcopy(self)
        temp ^= other
        return temp

    def __iadd__(self, other):
        func_ = self.func
        if isinstance(other, (int, float)):
            self.func = lambda x : func_(x) + other
        elif isinstance(other, Map):
            self.func = lambda x: func_(x) + other.func(x)
        else:
            raise TypeError("Cannot add Map and object of type " + str(type(other)))
        return self

    def __add__(self, other : 'Map') -> 'Map':
        """
        Returns the pointwise sum of two maps, given by

        .. math::

            (f + g)(t) := f(t) + g(t)

        :type other: Map
        :rtype: Map

        Example usage::

            import kauri.bck as bck

            m1 = 2 * bck.antipode
            m2 = bck.antipode + bck.antipode # Same as m1
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
    __rxor__ = __xor__
    __radd__ = __add__
    __rsub__ = __sub__

    def __and__(self, other : 'Map') -> 'Map': #TODO: scalars?
        """
        Returns the composition of two maps, given by

        .. math::

            (f \\, \\& \\, g)(t) := (f \\circ g)(t) := f(g(t))

        :type other: Map
        :rtype: Map

        Example usage::

            import kauri as kr
            import kauri.bck as bck

            t = kr.Tree([[]])

            (bck.antipode & bck.antipode)(t)
            bck.antipode(bck.antipode(t)) #Same as above
        """
        return Map(lambda x : self(other(x) * Tree(None)))

    def modified_equation(self) -> 'Map':
        """
        Assuming the given map :math:`\\phi` corresponds to the elementary weights
        function of a B-series method, returns the map corresponding to the elementary
        weights function of the modified (B-series) vector field, :math:`\\widetilde{\\phi}`,
        defined by

        .. math::

            (\\widetilde{\\phi} \\star e)(t) = \\phi(t)

        where :math:`e(t) = 1 / t!` is the elementary weights function of
        the exact solution, or equivalently

        .. math::

            \\widetilde{\\phi}(t) = (\\phi \\star e^{\\star (-1)})(t)

        where :math:`\\mathrm{Id}` is the identity map on trees and
        :math:`e^{\\star (-1)} = e \\circ S_{CEM}` :cite:`chartier2010algebraic`.

        :return: Elementary weights map of the modified vector field
        """
        return self.log()

    def preprocessed_integrator(self) -> 'Map':
        #TODO
        return exact_weights ^ (self & Map(cem_antipode))

    def exp(self) -> 'Map':
        """
        Returns the exponential of the map, defined as

        .. math::

            \\exp(\\phi) = \\phi \\star e

        where :math:`e(t) = 1 / t!` is the elementary weights function
        of the exact solution :cite:`chartier2010algebraic, murua2006hopf`.

        :return: Exponential map
        :rtype: Map
        """
        return self ^ exact_weights

    def log(self) -> 'Map':
        """
        Returns the logarithm of the map, defined as

        .. math::

            \\log(\\phi) = \\phi \\star e^{\\star (-1)}

        where :math:`e(t) = 1 / t!` is the elementary weights function
        of the exact solution and :math:`e^{\\star (-1)} = e \\circ S_{CEM}`
        :cite:`chartier2010algebraic, murua2006hopf`.

        :return: Exponential map
        :rtype: Map
        """
        return self ^ (exact_weights & Map(cem_antipode))


# Some common examples provided for convenience
ident = Map(lambda x : x)
sign = Map(lambda x : x.sign())
exact_weights = Map(lambda x : 1. / x.factorial())

omega = Map(lambda x : 1 if (x == Tree(None) or x == Tree([])) else 0).log()
