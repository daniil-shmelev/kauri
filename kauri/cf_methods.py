# Copyright 2026 Daniil Shmelev
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
Named commutator-free (CF) Lie group integrators.

Every method is a :class:`~kauri.cf.CFMethod` instance with a published
Butcher-like tableau.  Use ``method.lb_character()`` for the numerical
Lie-Butcher character and ``method.symbolic_lb_character()`` for the
same character expressed in sympy rationals.

``lie_euler``, ``lie_midpoint``, ``cfree_rk3`` and ``cfree_rk4`` follow
the classical RKMK family with a single exponential per step
(``J = 1``); their LB characters coincide with the elementary weights of
the underlying Runge--Kutta method on planar trees.  These are the
order-1, 2, 3 and 4 "base" commutator-free methods that fit the
single-exponential-per-stage structure of :class:`CFMethod`.

The genuinely multi-exponential schemes introduced in Celledoni,
Marthinsen and Owren (2003) "Commutator-free Lie group methods" rely on
flow reuse across stages (e.g. :math:`Y_4 = \\exp(k_3 - \\tfrac{1}{2}k_1)
\\circ Y_2`), which the current :class:`CFMethod` API does not model:
every stage is assumed to use a single exponential applied to
:math:`y_n`.  Users who need those methods should construct a bespoke
:class:`CFMethod` with a larger stage count or wait for multi-exponential
stage support.
"""
from fractions import Fraction as _F

from .cf import CFMethod

# ---------------------------------------------------------------------------
# J = 1 ("RKMK") instances
# ---------------------------------------------------------------------------

lie_euler = CFMethod(
    a=[[_F(0)]],
    betas=[[_F(1)]],
    name="Lie-Euler",
)
lie_euler.__doc__ = """
Lie-Euler method: ``y_{n+1} = exp(h f(y_n)) . y_n``.

One stage, one exponential (J = 1).  Planar order 1.
"""

lie_midpoint = CFMethod(
    a=[[_F(0), _F(0)],
       [_F(1, 2), _F(0)]],
    betas=[[_F(0), _F(1)]],
    name="Lie-Midpoint",
)
lie_midpoint.__doc__ = """
Implicit Lie-midpoint in its explicit RKMK form: evaluate ``f`` at a
half-step and take one full-step exponential.

Two stages, one exponential (J = 1).  Planar order 2.
"""

cfree_rk3 = CFMethod(
    # Kutta's third-order method
    a=[[_F(0), _F(0), _F(0)],
       [_F(1, 2), _F(0), _F(0)],
       [_F(-1), _F(2), _F(0)]],
    betas=[[_F(1, 6), _F(2, 3), _F(1, 6)]],
    name="CFree-RK3",
)
cfree_rk3.__doc__ = """
RKMK variant of Kutta's third-order Runge--Kutta method.

Three stages, one exponential (J = 1).  Planar order 3.
"""

cfree_rk4 = CFMethod(
    # Classical fourth-order Runge--Kutta tableau
    a=[[_F(0), _F(0), _F(0), _F(0)],
       [_F(1, 2), _F(0), _F(0), _F(0)],
       [_F(0), _F(1, 2), _F(0), _F(0)],
       [_F(0), _F(0), _F(1), _F(0)]],
    betas=[[_F(1, 6), _F(1, 3), _F(1, 3), _F(1, 6)]],
    name="CFree-RK4",
)
cfree_rk4.__doc__ = """
RKMK variant of the classical fourth-order Runge--Kutta method.

Four stages, one exponential (J = 1).  Planar order 4.
"""


