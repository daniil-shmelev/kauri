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
We provide a handful of Runge-Kutta schemes for convenience, as instances of the :class:`RK` class.
We list these below, along with their Butcher tableux.
"""

from math import sqrt

from .rk import RK

euler = RK([[0]], [1], 'Euler')
euler.__doc__ = """
The Euler method

.. math::

        \\begin{array}{c|c}
            0 & 0 \\\\
            \\hline
             & 1
        \\end{array}
"""

heun_rk2 = RK([[0, 0],
            [1, 0]],
        [0.5, 0.5], 'Heun RK2')
heun_rk2.__doc__ = """
Heun's RK2 method

.. math::

        \\begin{array}{c|cc}
            0 & 0 & 0 \\\\
            1 & 1 & 0 \\\\
            \\hline
             & 1/2 & 1/2
        \\end{array}
"""

midpoint = RK([[0, 0],
                  [0.5, 0]],
              [0, 1], 'Midpoint')
midpoint.__doc__ = """
The midpoint method

.. math::

        \\begin{array}{c|cc}
            0 & 0 & 0 \\\\
            1/2 & 1/2 & 0 \\\\
            \\hline
             & 0 & 1
        \\end{array}
"""

kutta_rk3 = RK([[0, 0, 0],
                         [0.5, 0, 0],
                         [-1, 2, 0]],
               [1/6, 2/3, 1/6], 'Kutta RK3')
kutta_rk3.__doc__ = """
Kutta's RK3 method

.. math::

        \\begin{array}{c|ccc}
            0 & 0 & 0 & 0 \\\\
            1/2 & 1/2 & 0 & 0 \\\\
            1 & -1 & 2 & 0 \\\\
            \\hline
             & 1/6 & 2/3 & 1/6
        \\end{array}
"""

heun_rk3 = RK([[0, 0, 0],
                        [1/3, 0, 0],
                        [0, 2/3, 0]],
              [1/4, 0, 3/4], 'Heun RK3')
heun_rk3.__doc__ = """
Heun's RK3 method

.. math::

        \\begin{array}{c|ccc}
            0 & 0 & 0 & 0 \\\\
            1/3 & 1/3 & 0 & 0 \\\\
            2/3 & 0 & 2/3 & 0 \\\\
            \\hline
             & 1/4 & 0 & 3/4
        \\end{array}
"""

# Ralston's Third-Order Method (RK3)
ralston_rk3 = RK([[0, 0, 0],
                           [1/2, 0, 0],
                           [0, 3/4, 0]],
                 [2/9, 1/3, 4/9], 'Ralston RK3')
ralston_rk3.__doc__ = """
Ralston's RK3 method

.. math::

        \\begin{array}{c|ccc}
            0 & 0 & 0 & 0 \\\\
            1/2 & 1/2 & 0 & 0 \\\\
            3/4 & 0 & 3/4 & 0 \\\\
            \\hline
             & 2/9 & 1/3 & 4/9
        \\end{array}
"""

rk4 = RK([[0, 0, 0, 0],
                           [0.5, 0, 0, 0],
                           [0, 0.5, 0, 0],
                           [0, 0, 1, 0]],
                 [1/6, 1/3, 1/3, 1/6], 'RK4')
rk4.__doc__ = """
The RK4 method

.. math::

        \\begin{array}{c|cccc}
            0 & 0 & 0 & 0 & 0 \\\\
            1/2 & 1/2 & 0 & 0 & 0 \\\\
            1/2 & 0 & 1/2 & 0 & 0 \\\\
            1 & 0 & 0 & 1 & 0 \\\\
            \\hline
             & 1/6 & 1/3 & 1/3 & 1/6
        \\end{array}
"""

ralston_rk4 = RK([[0, 0, 0, 0],
                           [2/5, 0, 0, 0],
                           [(-2889 + 1428*sqrt(5)) / 1024, (3785 - 1620*sqrt(5)) / 1024, 0, 0],
                           [(-3365 + 2094*sqrt(5))/6040, (-975-3046*sqrt(5))/2552, (467040 + 203968*sqrt(5)) / 240845, 0]],
                 [(263 + 24*sqrt(5)) / 1812, (125 - 1000*sqrt(5))/3828, (3426304 + 1661952*sqrt(5)) / 5924787, (30 - 4*sqrt(5)) / 123], 'Ralston RK4')
ralston_rk4.__doc__ = """
Ralston's RK4 method

.. math::

    \\begin{array}{c|cccc}
        0 & 0 & 0 & 0 & 0 \\\\
        \\frac{2}{5} & \\frac{2}{5} & 0 & 0 & 0 \\\\
        \\frac{14 - 3\\sqrt{5}}{16} & \\frac{-2889 + 1428\\sqrt{5}}{1024} & \\frac{3785 - 1620\\sqrt{5}}{1024} & 0 & 0 \\\\
        1 & \\frac{-3365 + 2094\\sqrt{5}}{6040} & \\frac{-975-3046\\sqrt{5}}{2552} & \\frac{467040+203968\\sqrt{5}}{240845} & 0 \\\\
        \\hline
        & \\frac{263+24\\sqrt{5}}{1812} & \\frac{125 - 1000\\sqrt{5}}{3828} & \\frac{3426304+1661952\\sqrt{5}}{5924787} & \\frac{30-4\\sqrt{5}}{123}
    \\end{array}

"""

nystrom_rk5 = RK([[0,0,0,0,0,0],
                  [1/3,0,0,0,0,0],
                  [4/25, 6/25, 0,0,0,0],
                  [1/4, -3, 15/4, 0,0,0],
                  [2/27, 10/9, -50/81, 8/81, 0,0],
                  [2/25, 12/25, 2/15, 8/75, 0,0]],
                 [23/192, 0, 125/192, 0, -27/64, 125/192], 'Nystrom RK5')
nystrom_rk5.__doc__ = """
Nyström's RK5 method

.. math::

    \\begin{array}{c|cccccc}
        0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        1/3 & 1/3 & 0 & 0 & 0 & 0 & 0 \\\\
        2/5 & 4/25 & 6/25 & 0 & 0 & 0 & 0 \\\\
        1 & 1/4 & -3 & 15/4 & 0 & 0 & 0 \\\\
        2/5 & 2/27 & 10/9 & -50/81 & 8/81 & 0 & 0 \\\\
        4/5 & 2/25 & 12/25 & 2/15 & 8/75 & 0 & 0 \\\\
        \\hline
        & 23/192 & 0 & 125/192 & 0 & -27/64 & 125/192
    \\end{array}

"""

#######################################################################################
################################# Implicit Methods ####################################
#######################################################################################

backward_euler = RK([[1]], [1], 'Backward Euler')
backward_euler.__doc__ = """
The backward Euler method

.. math::

        \\begin{array}{c|c}
            1 & 1 \\\\
            \\hline
             & 1
        \\end{array}
"""

implicit_midpoint = RK([[0.5]], [1], 'Implicit Midpoint')
implicit_midpoint.__doc__ = """
The implicit midpoint method

.. math::

        \\begin{array}{c|c}
            1/2 & 1/2 \\\\
            \\hline
             & 1
        \\end{array}
"""

crank_nicolson = RK([[0, 0],
                                  [0.5, 0.5]],
                        [0.5, 0.5], 'Crank Nicolson')
crank_nicolson.__doc__ = """
The Crank Nicolson method

.. math::

        \\begin{array}{c|cc}
            0 & 0 & 0 \\\\
            1 & 1/2 & 1/2 \\\\
            \\hline
             & 1/2 & 1/2
        \\end{array}
"""

gauss6 = RK([
    [5/36, 2/9 - sqrt(15) / 15, 5/36 - sqrt(15) / 30],
    [5/36 + sqrt(15)/24, 2/9, 5/36 - sqrt(15)/24],
    [5/36 + sqrt(15)/30, 2/9 + sqrt(15)/15, 5/36]
               ],
              [5/18, 4/9, 5/18], 'Gauss 6')
gauss6.__doc__ = """
Kuntzmann & Butcher method of order 6, based on Gaussian quadrature

.. math::

    \\begin{array}{c|ccc}
        \\frac{1}{2} - \\frac{\\sqrt{15}}{10} & \\frac{5}{36} & \\frac{2}{9} - \\frac{\\sqrt{15}}{15} & \\frac{5}{36} - \\frac{\\sqrt{15}}{30} \\\\
        \\frac{1}{2} & \\frac{5}{36} + \\frac{\\sqrt{15}}{24} & \\frac{2}{9} & \\frac{5}{36} - \\frac{\\sqrt{15}}{24} \\\\
        \\frac{1}{2} - \\frac{\\sqrt{15}}{10} & \\frac{5}{36} + \\frac{\\sqrt{15}}{30} & \\frac{2}{9} + \\frac{\\sqrt{15}}{15} & \\frac{5}{36} \\\\
        \\hline
         & \\frac{5}{18} & \\frac{4}{9} & \\frac{5}{18}
    \\end{array}

"""

radau_iia = RK([
    [(88 - 7 * sqrt(6)) / 360, (296 - 169 * sqrt(6)) / 1800, (-2 + 3 * sqrt(6)) / 225],
    [(296 + 169 * sqrt(6)) / 1800, (88 + 7 * sqrt(6)) / 360, (-2 - 3 * sqrt(6)) / 225],
    [(16 - sqrt(6)) / 36, (16 + sqrt(6)) / 36, 1/9]
               ],
              [(16 - sqrt(6)) / 36, (16 + sqrt(6)) / 36, 1/9], 'Radau IIA')
radau_iia.__doc__ = """
The Radau IIA method of order 5

.. math::

    \\begin{array}{c|ccc}
        \\frac{4 - \\sqrt{6}}{10} & \\frac{88 - 7\\sqrt{6}}{360} & \\frac{296 - 169\\sqrt{6}}{1800} & \\frac{-2 + 3\\sqrt{6}}{225} \\\\
        \\frac{4 + \\sqrt{6}}{10} & \\frac{296 + 169\\sqrt{6}}{1800} & \\frac{88 + 7\\sqrt{6}}{360} & \\frac{-2 - 3\\sqrt{6}}{225} \\\\
        1 & \\frac{16 - \\sqrt{6}}{36} & \\frac{16 + \\sqrt{6}}{36} & \\frac{1}{9} \\\\
        \\hline
         & \\frac{16 - \\sqrt{6}}{36} & \\frac{16 + \\sqrt{6}}{36} & \\frac{1}{9} \\\\
    \\end{array}

"""

lobatto6 = RK([
    [0,0,0,0],
    [(5 + sqrt(5)) / 60, 1/6, (15 - 7 * sqrt(5)) / 60, 0],
    [(5 - sqrt(5)) / 60, (15 + 7 * sqrt(5)) / 60, 1/6, 0],
    [1/6, (5-sqrt(5)) / 12, (5 + sqrt(5))/12, 0]
               ],
              [1/12, 5/12, 5/12, 1/12], 'Lobatto 6')
lobatto6.__doc__ = """
Butcher’s Lobatto formula of order 6

.. math::

        \\begin{array}{c|cccc}
            0 & 0 & 0 & 0 & 0 \\\\
            \\frac{5 - \\sqrt{5}}{10} & \\frac{5 + \\sqrt{5}}{60} & \\frac{1}{6} & \\frac{15 - 7\\sqrt{5}}{60} & 0 \\\\
            \\frac{5 + \\sqrt{5}}{10} & \\frac{5 - \\sqrt{5}}{60} & \\frac{15 + 7\\sqrt{5}}{60} & \\frac{1}{6} & 0 \\\\
            1 & \\frac{1}{6} & \\frac{5 - \\sqrt{5}}{12} & \\frac{5 + \\sqrt{5}}{12} & 0 \\\\
            \\hline
             & \\frac{1}{12} & \\frac{5}{12} & \\frac{5}{12} & \\frac{1}{12}
        \\end{array}
"""

#######################################################################################
################################# EES Methods #########################################
#######################################################################################

def EES25(x):#TODO: Doc
    b1 = x
    b2 = 1/2
    b3 = 1/2 - x
    a21 = (1+2*x) / (4*(1-x))
    a31 = (4*x-1)**2 / (4*(x-1)*(1-4*x**2))
    a32 = (1-x) / (1-4*x**2)

    b = [b1, b2, b3]
    A = [[0,0,0],
        [a21, 0, 0],
        [a31, a32, 0]]
    return RK(A, b, 'EES25')


def EES27(x, plus=True):# TODO: doc
    sqrt2 = sqrt(2) if plus else - sqrt(2)

    b1 = x
    b2 = 0.5 * (2 - sqrt2) - (1 - sqrt2) * x
    b3 = (1 - sqrt2) * (x - 1)
    b4 = 0.5 * (2 - sqrt2) - x

    alpha = (-2 * x + sqrt2 + 1) * (2 * x + sqrt2) / ((2 * x - 1) * (4 * x ** 2 - 4 * x - 1))
    beta = (1 + sqrt2 - 2 * x) * (2 + sqrt2 - 2 * x) / (
                (2 * x - 1) * (2 * x ** 2 - 4 * x + 1) * (4 * x ** 2 - 4 * x - 1))

    a21 = (-2 + sqrt2 * (1 - 2 * x)) / (4 * (x - 1))
    a31 = (2 * x + sqrt2 - 2) * (4 * x + sqrt2 - 2) * alpha / (4 * sqrt2 * (x - 1))
    a32 = 0.5 * (-1 + sqrt2) * alpha
    a41 = beta * (2 * x - sqrt2) * (-40 * x ** 4 + (80 - 40 * sqrt2) * x ** 3 - (88 - 60 * sqrt2) * x ** 2 + (
                48 - 34 * sqrt2) * x + 7 * sqrt2 - 10) / (8 * (x - 1) * (2 * x ** 2 - 1))
    a42 = 0.5 * (2 - sqrt2) * x * (x - 1) * (4 * x + sqrt2 - 2) * beta
    a43 = (2 - sqrt2) * (2 * x - sqrt2) * (2 + sqrt2 - 2 * x) * (x - 1) * (2 * x - 1) / (
                4 * (2 * x ** 2 - 1) * (2 * x ** 2 - 4 * x + 1))

    b = [b1, b2, b3, b4]
    A = [[0, 0, 0, 0],
         [a21, 0, 0, 0],
         [a31, a32, 0, 0],
         [a41, a42, a43, 0]]

    return RK(A, b, 'EES27')