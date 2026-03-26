# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
This module provides the :class:`Map` class, which implements linear multiplicative maps on trees
and allows for their manipulation with respect to different Hopf algebras. In particular, this covers
characters on the Hopf algebra, as well as more complicated maps.
"""
import copy
from typing import Union, Callable

from .trees import (Tree, PlanarTree, Forest, NoncommutativeForest, ForestSum,
                     TensorProductSum, EMPTY_TREE, _is_scalar, _is_planar_obj)
from ._protocols import TreeLike, ForestLike, ForestSumLike
from .generic_algebra import apply_map, func_power, func_product

class Map:
    """
    A multiplicative linear map on rooted trees. This class is callable.

    When applied to a forest, the map is extended multiplicatively:
    ``f(t1 * t2 * ... * tk) = f(t1) * f(t2) * ... * f(tk)``.

    If ``anti=True``, the map is extended as an anti-homomorphism instead:
    ``f(t1 * t2 * ... * tk) = f(tk) * ... * f(t2) * f(t1)``.
    This is required for antipodes of noncommutative Hopf algebras (e.g. PBCK, PGL).

    :param func: A function taking as input a single tree and returning a scalar,
        Tree, Forest or ForestSum.
    :type func: Callable[[Tree], int | float | Tree | Forest | ForestSum]
    :param anti: If True, extend to forests as an anti-homomorphism (reversed order).
        Default is False.
    :type anti: bool
    """
    def __init__(self, func : Callable[[Tree], Union[int, float, Tree, Forest, ForestSum]], anti=False):
        if not callable(func):
            raise TypeError("func parameter must be callable")
        self.func = func
        self.anti = anti
        self._cache = {}

    def __call__(self, t : Union[Tree, Forest, ForestSum]) -> Union[int, float, Tree, Forest, ForestSum]:
        """Applies the map to a tree, forest, or forest sum (extends linearly and multiplicatively)."""
        if isinstance(t, TensorProductSum):
            raise TypeError("Cannot apply Map to TensorProductSum. "
                            "Apply the map to each tensor factor separately.")
        if not isinstance(t, (TreeLike, ForestLike, ForestSumLike)):
            raise TypeError("Argument to Map must be Tree, Forest or ForestSum, not " + str(type(t)))
        try:
            return self._cache[t]
        except KeyError:
            result = apply_map(t, self.func, anti=self.anti)
            self._cache[t] = result
            return result

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

        from .bck.bck import coproduct_impl as bck_coproduct, counit_impl as bck_counit, antipode_impl as bck_antipode
        return Map(lambda x : func_power(x, self.func, exponent, bck_coproduct, bck_counit, bck_antipode))

    def __imul__(self, other : Union[int, float, 'Map']):
        func_ = self.func
        if _is_scalar(other):
            self.func = lambda x : other * func_(x)
        elif isinstance(other, Map):
            from .bck.bck import coproduct_impl as bck_coproduct
            self.func = lambda x : func_product(x, func_, other.func, bck_coproduct)
        else:
            raise TypeError("Error in BCK product: Cannot multiply Map by object of type " + str(type(other)))
        self._cache.clear()
        return self

    def __ixor__(self, other):
        func_ = self.func
        if _is_scalar(other):
            self.func = lambda x: other * func_(x)
        elif isinstance(other, Map):
            from .cem.cem import _coproduct_raw as cem_coproduct
            def f_(x):
                if x.list_repr is None:
                    out = other.func(EMPTY_TREE)
                else:
                    out = func_product(x, func_, other.func, cem_coproduct)
                if isinstance(out, (Forest, NoncommutativeForest, ForestSum)):
                    out = out.singleton_reduced()
                    if isinstance(out, ForestSum):
                        out = out.simplify()
                return out
            self.func = f_
        else:
            raise TypeError("Error in CEM product: Cannot multiply Map by object of type " + str(type(other)))
        self._cache.clear()
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
        if _is_scalar(other):
            self.func = lambda x : func_(x) + other
        elif isinstance(other, Map):
            self.func = lambda x: func_(x) + other.func(x)
        else:
            raise TypeError("Cannot add Map and object of type " + str(type(other)))
        self._cache.clear()
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
        return Map(lambda x : -self.func(x), anti=self.anti)

    def __isub__(self, other):
        self.__iadd__(-other)
        return self

    def __sub__(self, other):
        """Returns the pointwise difference of two maps: ``(f - g)(t) = f(t) - g(t)``."""
        temp = copy.deepcopy(self)
        temp -= other
        return temp

    def __rsub__(self, other):
        return (-self) + other

    __rmul__ = __mul__
    __rxor__ = __xor__
    __radd__ = __add__

    def __and__(self, other : 'Map') -> 'Map':
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
        if not isinstance(other, Map):
            raise TypeError("Cannot compose Map with object of type " + str(type(other)))
        def _composed(x):
            inner = other(x)
            empty = PlanarTree(None) if _is_planar_obj(inner) else Tree(None)
            return self(inner * empty)
        return Map(_composed)

    def modified_equation(self) -> 'Map':
        """
        Assuming the given map :math:`\\phi` corresponds to the elementary weights
        function of a B-series method, returns the map corresponding to the coefficients
        of the modified (B-series) vector field, :math:`\\widetilde{\\phi}`,
        defined by

        .. math::

            (\\widetilde{\\phi} \\star e)(t) = \\phi(t)

        where :math:`e(t) = 1 / t!` is the elementary weights function of
        the exact solution, or equivalently

        .. math::

            \\widetilde{\\phi}(t) = (\\phi \\star e^{\\star (-1)})(t)

        and :math:`e^{\\star (-1)} = e \\circ S_{CEM}` :cite:`chartier2010algebraic`.

        :return: Map corresponding to the modified vector field
        """
        return self.log()

    def preprocessed_integrator(self) -> 'Map':
        """
        Assuming the given map :math:`\\phi` corresponds to the elementary weights
        function of a B-series method, returns the map corresponding to the
        preprocessed integrator, :math:`\\widetilde{\\phi}`, defined by

        .. math::

            (\\widetilde{\\phi} \\star \\phi)(t) = e(t)

        where :math:`e(t) = 1 / t!` is the elementary weights function of
        the exact solution, or equivalently

        .. math::

            \\widetilde{\\phi}(t) = (e \\star \\phi^{\\star (-1)})(t)

        and :math:`\\phi^{\\star (-1)} = \\phi \\circ S_{CEM}` :cite:`chartier2010algebraic`.

        :return: Map corresponding to the preprocessed integrator
        """
        from .cem.cem import antipode_impl as cem_antipode
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

        :return: Logarithm map
        :rtype: Map
        """
        from .cem.cem import antipode_impl as cem_antipode
        return self ^ (exact_weights & Map(cem_antipode))


# Some common examples provided for convenience
ident = Map(lambda x : x)
ident.__doc__ = """
The identity map, :math:`t \\mapsto t`.
"""
sign = Map(lambda x : x.sign())
sign.__doc__ = """
The sign map, or canonical involution, :math:`t \\mapsto (-1)^{|t|} t`.
"""
exact_weights = Map(lambda x : 1. / x.factorial())
exact_weights.__doc__ = """
The elementary weights function of the exact solution, :math:`t \\mapsto 1/t!`.
"""
omega = Map(lambda x : 1 if (x == Tree(None) or x == Tree([])) else 0).log()
omega.__doc__ = """
The coefficients of the modified equation for the (explicit) Euler method,
:math:`t \\mapsto \\omega(t) := \\log(\\delta_\\emptyset + \\delta_\\bullet)`. 
See :cite:`chartier2010algebraic` for details.
"""
