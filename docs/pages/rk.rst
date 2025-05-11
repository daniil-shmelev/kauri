Runge--Kutta Schemes
========================

.. automodule:: kauri.rk

.. autofunction:: kauri.rk.rk_symbolic_weight
.. autofunction:: kauri.rk.rk_order_cond
.. autoclass:: kauri.rk.RK
   :members:
   :special-members: __mul__, __pow__, __add__


Instances
---------------

.. automodule:: kauri.rk_methods

Explicit Methods
~~~~~~~~~~~~~~~~~~

.. autodata:: euler
    :no-value:
.. autodata:: heun_rk2
    :no-value:
.. autodata:: midpoint
    :no-value:
.. autodata:: kutta_rk3
    :no-value:
.. autodata:: heun_rk3
    :no-value:
.. autodata:: ralston_rk3
    :no-value:
.. autodata:: rk4
    :no-value:
.. autodata:: ralston_rk4
    :no-value:
.. autodata:: nystrom_rk5
    :no-value:

Implicit Methods
~~~~~~~~~~~~~~~~~~

.. autodata:: backward_euler
    :no-value:
.. autodata:: implicit_midpoint
    :no-value:
.. autodata:: crank_nicolson
    :no-value:
.. autodata:: gauss6
    :no-value:
.. autodata:: radau_iia
    :no-value:
.. autodata:: lobatto6
    :no-value:

.. include:: ../refs.rst