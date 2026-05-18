Commutator-Free Methods
========================

.. automodule:: kauri.cf

Lie-Butcher substitution
------------------------

Kauri represents Lie-Butcher (LB) substitution and composition through
MKW convolution of basis-aware :class:`~kauri.maps.Map` objects.  A
commutator-free method with exponentials applied right-to-left has

.. math::

   \alpha = \alpha_J *_\mathrm{MKW} \cdots *_\mathrm{MKW} \alpha_1,

where each :math:`\alpha_l` is the elementary-weight character of the
Runge--Kutta method ``(A, beta_l)``.

Use :meth:`CFMethod.lb_character` for the numerical LB character and
:meth:`CFMethod.symbolic_lb_character` for an exact symbolic character.
Both return :class:`~kauri.maps.Map` instances using the MKW
``extension="shuffle"`` convention.

.. autoclass:: kauri.cf.CFMethod
   :members:

.. autoclass:: kauri.cf.ReusedStageCFMethod
   :members:

Named methods
-------------

The :mod:`kauri.cf_methods` module provides named commutator-free
methods as ready-to-use :class:`CFMethod` instances.

.. automodule:: kauri.cf_methods

.. autodata:: kauri.cf_methods.lie_euler
   :no-value:

.. autodata:: kauri.cf_methods.lie_midpoint
   :no-value:

.. autodata:: kauri.cf_methods.cfree_rk3
   :no-value:

.. autodata:: kauri.cf_methods.cfree_rk4
   :no-value:
