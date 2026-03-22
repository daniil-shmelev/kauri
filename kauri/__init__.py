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
from .trees import Tree, Forest, ForestSum, TensorProductSum
from .maps import Map, ident, sign, exact_weights, omega
from .display import display
from .gentrees import trees_of_order, trees_up_to_order
from .rk import RK, rk_symbolic_weight, rk_order_cond

from .rk_methods import (euler, heun_rk2, midpoint, kutta_rk3, heun_rk3,
                         ralston_rk3, rk4, ralston_rk4, nystrom_rk5, backward_euler,
                         implicit_midpoint, crank_nicolson, gauss6, radau_iia, lobatto6,
                         EES25, EES27)

from .bseries import BSeries, elementary_differential

from .trees import EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM, ZERO_FOREST_SUM

import kauri.bck
import kauri.cem

from .oddeven import id_sqrt, minus, plus