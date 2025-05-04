"""
Algebraic manipulation of rooted trees for the analysis of B-series and Runge-Kutta schemes.
"""
from .trees import Tree, Forest, ForestSum
from .maps import Map, ident, sign, exact_weights, omega
from .display import display
from .gentrees import trees_of_order, trees_up_to_order
from .rk import RK, rk_symbolic_weight, rk_order_cond
from .rk_methods import *
from .bseries import BSeries, elementary_differential

from .trees import EMPTY_TREE, EMPTY_FOREST, EMPTY_FOREST_SUM, ZERO_FOREST_SUM

import kauri.bck
import kauri.cem