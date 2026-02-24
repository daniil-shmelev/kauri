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
Algebraic manipulation of rooted trees for the analysis of B-series and Runge-Kutta schemes.
"""

import kauri.bck
import kauri.cem

from .bseries import BSeries, elementary_differential
from .display import display
from .gentrees import trees_of_order, trees_up_to_order
from .maps import Map, exact_weights, ident, omega, sign
from .oddeven import id_sqrt, minus, plus
from .rk import RK, rk_order_cond, rk_symbolic_weight
from .rk_methods import (
    EES25,
    EES27,
    backward_euler,
    crank_nicolson,
    euler,
    gauss6,
    heun_rk2,
    heun_rk3,
    implicit_midpoint,
    kutta_rk3,
    lobatto6,
    midpoint,
    nystrom_rk5,
    radau_iia,
    ralston_rk3,
    ralston_rk4,
    rk4,
)
from .trees import (
    EMPTY_FOREST,
    EMPTY_FOREST_SUM,
    EMPTY_TREE,
    ZERO_FOREST_SUM,
    Forest,
    ForestSum,
    TensorProductSum,
    Tree,
)
