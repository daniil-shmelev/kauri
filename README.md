<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/logo_light.png" width="350" alt="Logo">
  </picture>
</p>

![PyPI - Version](https://img.shields.io/pypi/v/kauri)
![PyPI - Downloads](https://img.shields.io/pypi/dm/kauri)
![CI - Test](https://github.com/daniil-shmelev/kauri/actions/workflows/tests.yml/badge.svg)
![Read the Docs](https://img.shields.io/readthedocs/kauri)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Kauri is a Python package for symbolic and algebraic manipulation of rooted trees, with applications to B-series, Runge-Kutta methods, and backward error analysis. It implements multiple Hopf algebraic structures on both non-planar and planar rooted trees, and provides tools for symbolic computation, visualization, and numerical integration.

## Installation

```
pip install kauri
```

## Documentation

Full documentation is available at [https://kauri.readthedocs.io](https://kauri.readthedocs.io)

## Features

### Hopf algebras

| Algebra | Non-planar | Planar |
|---------|-----------|--------|
| Butcher-Connes-Kreimer (BCK) | `kauri.bck` | `kauri.nck` |
| Grossman-Larson (GL) | `kauri.gl` | `kauri.pgl` |
| Calaque-Ebrahimi-Fard-Manchon (CEM) | `kauri.cem` | -- |
| Munthe-Kaas-Wright (MKW) | -- | `kauri.mkw` |

Each algebra provides: `coproduct`, `counit`, `antipode`, `map_product`, `map_power`.
Additionally, `kauri.gl` and `kauri.pgl` provide `product`, and `kauri.mkw` provides `shuffle_product`.

### Tree types

| Non-planar (commutative) | Planar (noncommutative) |
|-------------------------|------------------------|
| `Tree` | `PlanarTree` |
| `Forest` | `OrderedForest` |
| `ForestSum` | `ForestSum` |
| `TensorProductSum` | `TensorProductSum` |

### Additional modules

- **B-series** (`BSeries`) -- symbolic and numerical manipulation of truncated B-series
- **Runge-Kutta methods** (`RK`) -- 15+ predefined methods with order verification, composition, and numerical integration
- **Commutator-free methods** (`CFMethod`) -- Lie group integrators with planar order theory
- **Odd-even decomposition** (`oddeven`, `planar_oddeven`) -- symmetric splitting of characters
- **Map algebra** (`Map`) -- BCK/CEM convolution products, composition, exp/log
- **Tree generation** -- enumeration of non-planar, planar, and coloured trees
- **SVG display** -- inline visualization in Jupyter notebooks

## Examples

### BCK coproduct

```python
import kauri as kr
import kauri.bck as bck

t = kr.Tree([[], [[]]])
cp = bck.coproduct(t)
kr.display(cp)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_bck_coproduct.svg" width="495" alt="BCK coproduct example">
</picture>

### Labelled BCK antipode

```python
t = kr.Tree([[[3],2],[1],0])
s = bck.antipode(t)
kr.display(s)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_bck_antipode.svg" width="558" alt="BCK antipode example">
</picture>

### Grossman-Larson coproduct

```python
import kauri.gl as gl

t = kr.Tree([[], [[]]])
cp = gl.coproduct(t)
kr.display(cp)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_gl_coproduct.svg" width="266" alt="GL coproduct example">
</picture>

### CEM coproduct

```python
import kauri.cem as cem

t = kr.Tree([[], [[]]])
cp = cem.coproduct(t)
kr.display(cp)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_cem_coproduct.svg" width="551" alt="CEM coproduct example">
</picture>

### NCK coproduct

```python
import kauri.nck as nck

pt = kr.PlanarTree([[], [[]]])
cp = nck.coproduct(pt)
kr.display(cp)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_nck_coproduct.svg" width="495" alt="NCK coproduct example">
</picture>

### PGL product

```python
import kauri.pgl as pgl

t1 = kr.PlanarTree([[]])
t2 = kr.PlanarTree([[], []])
p = pgl.product(t1, t2)
kr.display(p)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_pgl_product.svg" width="236" alt="PGL product example">
</picture>

### MKW coproduct

```python
import kauri.mkw as mkw

pt = kr.PlanarTree([[], [[]]])
cp = mkw.coproduct(pt)
kr.display(cp)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_mkw_coproduct.svg" width="480" alt="MKW coproduct example">
</picture>

### Trees of order 4

```python
for t in kr.trees_of_order(4):
    kr.display(t)
```

<picture>
  <img src="https://raw.githubusercontent.com/daniil-shmelev/kauri/main/docs/_static/example_trees_order4.svg" width="200" alt="Trees of order 4">
</picture>

### Runge-Kutta order conditions

```python
t = kr.Tree([[],[]])
print(kr.rk_order_cond(t, s=3, explicit=True))
```
```
a10**2*b1 + b2*(a20 + a21)**2 - 1/3
```

### Truncated B-series of RK4

```python
import sympy as sp

y1 = sp.symbols('y1')
y = sp.Matrix([y1])
f = sp.Matrix([y1 ** 2])

m = kr.rk4.elementary_weights_map()
bs = kr.BSeries(y, f, weights=m, order=5)
print(bs.series())
```

### Odd-even decomposition

```python
import kauri.oddeven as oddeven

# The square root of the identity in BCK convolution
sqrt_id = oddeven.id_sqrt

# Verify: sqrt_id * sqrt_id == identity
t = kr.Tree([[],[]])
print((sqrt_id * sqrt_id)(t))  # Same as kr.ident(t)
```

### Modified equations and preprocessing

```python
# Modified equation of a B-series method
phi = kr.rk4.elementary_weights_map()
mod_eq = phi.modified_equation()

# Preprocessed integrator
preprocessed = phi.preprocessed_integrator()
```

## Citation
If you found this library useful in your research, please consider citing:
```bibtex
@misc{shmelev2025ees,
  title={Explicit and Effectively Symmetric Runge-Kutta Methods},
  author={Shmelev, Daniil and Ebrahimi-Fard, Kurusch and Tapia, Nikolas and Salvi, Cristopher},
  journal={arXiv:2507.21006},
  year={2025}
}
```
