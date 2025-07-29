# Explicit and Effectively Symmetric Runge-Kutta Methods

This directory contains supporting code for the paper
_[Explicit and Effectively Symmetric Runge-Kutta Methods](https://arxiv.org/abs/2507.21006)_.

## Derivation of EES Schemes

The notebook `gen_order_conds.ipynb` precomputes the order conditions necessary to derive
EES schemes, as forest sums.

The notebook `order_conds_complexity.ipynb` compares the complexity of order conditions generated
naively with those generated using the odd-even decomposition.

The notebook `mathematica_order_conds.ipynb` converts the order conditions to mathematica code.

The mathematica notebooks `ees25.nb` and `ees27.nb` solve these order conditions.

The notebook `parameter_selection.ipynb` finds the optimal parameters
for EES25 and EES27 by minimising the next available order conditions.

## Tables

The notebook `tree_tables.ipynb` contains code generating two tables of trees found in the paper.
The first of these relates to the elementary weights function of the implicit midpoint method
and its adjoint. The second lists $\mathrm{Id}^{1/2}(\tau)$, $\tau^-$ and $\tau^+$ for $|\tau| \leq 4$.


## Plots and Numerical Experiments

The notebook `symmetric_decomp_euler.ipynb` contains code computing the decomposition of the
Euler method into its symmetric and antisymmetric components, evaluated on the ODE $dy/dt = 10y$,
$y_0=1$ with a step size of $h=0.15$. The even component is viewed as a correction term connecting 
the odd component with the original scheme.

The notebook `stability.ipynb` generates plots of stability regions and order stars for various
Runge-Kutta methods, including EES(2,5) and EES(2,7).

The notebook `inverse_square_attraction.ipynb` runs the example of inverse square law attraction
using the implicit midpoint scheme and various EES schemes.

The notebook `galactic_orbit.ipynb` runs the example of a galactic orbit
using classical Runge-Kutta methods and EES(2,7).

## Citation
```bibtex    
@misc{shmelev2025ees,
  title={Explicit and Effectively Symmetric Runge-Kutta Methods}, 
  author={Shmelev, Daniil and Ebrahimi-Fard, Kurusch and Tapia, Nikolas and Salvi, Cristopher},
  journal={arXiv:2507.21006},
  year={2025}
}
```