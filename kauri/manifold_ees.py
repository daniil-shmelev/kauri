"""
Symbolic order-condition generation for manifold EES methods.

Given a commutator-free (CF) format (number of stages *s*, number of
exponentials *J*, explicit flag), this module:

1. Creates symbolic SymPy parameters ``a[i][j]`` and ``beta[l][i]``.
2. Computes the LB-series character ``alpha(tau)`` for each ordered tree
   via NCK convolution of the individual exponential characters.
3. Generates forward conditions ``alpha(tau) - 1/gamma(tau) = 0``.
4. Generates antisymmetry conditions ``D(tau) = 0``.
5. Provides a thin Groebner-basis wrapper around ``sympy.groebner``.
"""
from __future__ import annotations

import sympy

from .trees import PlanarTree, EMPTY_PLANAR_TREE, OrderedForest
from ._protocols import ForestLike
from .gentrees import planar_trees_of_order
from .rk import _elementary_symbolic
from .mkw.mkw import coproduct_impl, forest_coproduct_impl
from .generic_algebra import sign_factor


def _extract_symbols(conditions):
    """Collect and sort free symbols from a list of SymPy conditions."""
    if not conditions:
        return []
    return sorted(
        set().union(*(c.free_symbols for c in conditions if c != 0)),
        key=str,
    )


# ---------------------------------------------------------------------------
# Symbolic parameter creation
# ---------------------------------------------------------------------------

def symbolic_cf_params(s: int, J: int, explicit: bool = True):
    """
    Create symbolic SymPy parameters for a CF method.

    :param s: Number of stages.
    :param J: Number of exponentials.
    :param explicit: If True, set ``a[i][j] = 0`` for ``j >= i``.
    :returns: ``(a, betas)`` where *a* is a SymPy Matrix and *betas*
              is a list of *J* lists of length *s*.
    """
    a = sympy.Matrix(s, s, lambda i, j: sympy.symbols(f'a{i}{j}'))
    if explicit:
        for i in range(s):
            for j in range(i, s):
                a[i, j] = 0
    betas = [[sympy.symbols(f'beta{l}{i}') for i in range(s)] for l in range(J)]
    return a, betas


# ---------------------------------------------------------------------------
# Symbolic coproduct convolution helper
# ---------------------------------------------------------------------------

def _shuffle_sym_value(x, tree_func):
    """Symbolic shuffle-symmetric evaluation of a basis-aware base character.

    For a tree ``x``: return ``tree_func(x)``.
    For a forest ``(t_1, ..., t_k)``: return ``prod tree_func(t_i) / k!``.
    This is the correct extension for a base (exponential) MKW character;
    it is NOT used for convolution results, which are evaluated by
    :func:`_sym_coproduct_eval` recursively through the paper's coproduct.
    """
    if isinstance(x, ForestLike):
        non_empty = [t for t in x.tree_list if t.list_repr is not None]
        if not non_empty:
            return sympy.sympify(tree_func(x.tree_list[0]))
        val = sympy.Integer(1)
        for t in non_empty:
            val *= sympy.sympify(tree_func(t))
        k = len(non_empty)
        if k > 1:
            val = val / sympy.factorial(k)
        return val
    return sympy.sympify(tree_func(x))


def _sym_coproduct_eval(x, left_fn, right_fn):
    """Symbolic MKW convolution ``(left_fn * right_fn)(x)``.

    Uses the paper's coproduct — ``coproduct_impl`` on trees and
    ``forest_coproduct_impl`` on ordered forests — and evaluates each
    side by calling ``left_fn`` / ``right_fn`` on the resulting basis
    element.  The ``*_fn`` arguments MUST accept both trees and forests
    (a "basis-aware" symbolic character); for base characters defined
    only on trees, wrap via :func:`_shuffle_sym_value`.

    This gives associative iterated convolution by coassociativity of
    the MKW coproduct.
    """
    # Trivial cases
    if isinstance(x, PlanarTree) and x.list_repr is None:
        return sympy.Integer(1)
    if isinstance(x, ForestLike):
        trees = [t for t in x.tree_list if t.list_repr is not None]
        if not trees:
            return sympy.sympify(left_fn(x.tree_list[0])) * sympy.sympify(right_fn(x.tree_list[0]))
        if len(trees) == 1:
            return _sym_coproduct_eval(trees[0], left_fn, right_fn)
        cp = forest_coproduct_impl(x)
    else:
        cp = coproduct_impl(x)

    result = sympy.Integer(0)
    for c, left, right in cp.term_list:
        left_val = sympy.sympify(left_fn(left))
        right_val = sympy.sympify(right_fn(right))
        result += c * left_val * right_val
    return sympy.expand(result)


# ---------------------------------------------------------------------------
# Symbolic LB-series character via NCK convolution
# ---------------------------------------------------------------------------

def symbolic_lb_character(
    tree: PlanarTree,
    a: sympy.Matrix,
    betas: list[list],
    s: int,
    J: int,
    *,
    _b_mats: list[sympy.Matrix] | None = None,
) -> sympy.Expr:
    """
    Compute the LB-series character ``alpha(tau)`` symbolically.

    Uses the MKW convolution:
    ``alpha = alpha_J *_MKW ... *_MKW alpha_1``,
    where each ``alpha_l`` is the elementary-weight character of the
    exponential ``(A, beta_l)``, viewed as a shuffle-multiplicative
    character on H_MKW.  Forest-extension uses the shuffle-symmetric
    ``1/k!`` convention.

    :param tree: An ordered (planar) tree.
    :param a: Symbolic *s* x *s* coefficient matrix.
    :param betas: *J* weight vectors (lists of length *s*).
    :param s: Number of stages.
    :param J: Number of exponentials.
    :param _b_mats: Pre-built weight matrices (internal optimisation).
    :returns: A SymPy expression in the CF parameters.
    """
    if _b_mats is None:
        _b_mats = [sympy.Matrix(1, s, lambda _, i: betas[l][i]) for l in range(J)]

    def _ew_tree(l):
        """Tree-only elementary weight for exponential l."""
        b_l = _b_mats[l]
        return lambda t: _elementary_symbolic(t.list_repr, a, b_l, s)

    # Wrap each exponential's tree function as a basis-aware base character
    # (shuffle-symmetric extension to forests).
    def _ew_basis(l):
        tree_fn = _ew_tree(l)
        return lambda x: _shuffle_sym_value(x, tree_fn)

    ew_funcs = [_ew_basis(l) for l in range(J)]

    if J == 1:
        return sympy.expand(ew_funcs[0](tree))

    # Iteratively convolve: result = alpha_1, then alpha_2 *_MKW result, ...
    current = ew_funcs[0]
    for l in range(1, J):
        prev, outer = current, ew_funcs[l]
        current = lambda t, _p=prev, _o=outer: _sym_coproduct_eval(t, _o, _p)

    return sympy.expand(current(tree))


# ---------------------------------------------------------------------------
# Cached alpha builder (shared by forward & antisymmetry conditions)
# ---------------------------------------------------------------------------

def _build_alpha_cache(max_order, a, betas, s, J):
    """Pre-compute alpha(tau) and sign(tau) for all ordered trees up to *max_order*."""
    b_mats = [sympy.Matrix(1, s, lambda _, i: betas[l][i]) for l in range(J)]
    cache: dict[tuple, sympy.Expr] = {}
    sign_cache: dict[tuple, int] = {}

    for n in range(max_order + 1):
        for tree in planar_trees_of_order(n):
            cache[tree.list_repr] = symbolic_lb_character(
                tree, a, betas, s, J, _b_mats=b_mats)
            sign_cache[tree.list_repr] = sign_factor(tree)
    return cache, sign_cache


# ---------------------------------------------------------------------------
# Order-condition generation
# ---------------------------------------------------------------------------

def forward_conditions(
    max_order: int,
    alpha_cache: dict[tuple, sympy.Expr],
) -> list[sympy.Expr]:
    """
    Forward order conditions: ``alpha(tau) - 1/gamma(tau) = 0``
    for all ordered trees of order ``1`` through ``max_order``.
    """
    conds = []
    for n in range(1, max_order + 1):
        for tree in planar_trees_of_order(n):
            alpha_val = alpha_cache[tree.list_repr]
            exact_val = sympy.Rational(1, tree.factorial())
            c = sympy.simplify(alpha_val - exact_val)
            if c != 0:
                conds.append(c)
    return conds


def antisymmetry_conditions(
    max_order: int,
    alpha_cache: dict[tuple, sympy.Expr],
    sign_cache: dict[tuple, int],
) -> list[sympy.Expr]:
    """
    Antisymmetry conditions: ``D(tau) = 0`` for all ordered trees of
    order ``1`` through ``max_order``, where
    ``D = (sign . alpha) *_nck alpha - epsilon``.
    """
    # Pre-compute sign * alpha for all cached trees
    sign_alpha_cache = {
        key: sign_cache[key] * val
        for key, val in alpha_cache.items()
    }

    # Tree-only lookups
    def _alpha_tree(t):
        return alpha_cache[t.list_repr]

    def _sign_alpha_tree(t):
        return sign_alpha_cache[t.list_repr]

    # Basis-aware wrappers (shuffle-symmetric extension on forests)
    def _alpha(x):
        return _shuffle_sym_value(x, _alpha_tree)

    def _sign_alpha(x):
        return _shuffle_sym_value(x, _sign_alpha_tree)

    conds = []
    for n in range(1, max_order + 1):
        for tree in planar_trees_of_order(n):
            d_val = _sym_coproduct_eval(tree, _sign_alpha, _alpha)
            c = sympy.simplify(d_val)   # epsilon(tree) = 0 for non-empty trees
            if c != 0:
                conds.append(c)
    return conds


def generate_conditions(
    forward_order: int,
    antisymmetric_order: int,
    s: int,
    J: int,
    explicit: bool = True,
) -> dict:
    """
    Generate the full polynomial system for a manifold-EES(p, q) method.

    :param forward_order: Target classical order *p*.
    :param antisymmetric_order: Target antisymmetric order *q*.
    :param s: Number of stages.
    :param J: Number of exponentials.
    :param explicit: If True, zero the upper triangle of *A*.
    :returns: dict with keys ``'forward'``, ``'antisymmetry'``, ``'all'``,
              ``'a'``, ``'betas'``, ``'symbols'``.
    """
    a, betas = symbolic_cf_params(s, J, explicit)

    # Compute all alpha values once, shared by both condition generators
    max_order = max(forward_order, antisymmetric_order)
    alpha_cache, sign_cache = _build_alpha_cache(max_order, a, betas, s, J)

    fc = forward_conditions(forward_order, alpha_cache)
    ac = antisymmetry_conditions(antisymmetric_order, alpha_cache, sign_cache)
    all_conds = fc + ac

    syms = _extract_symbols(all_conds)

    return {
        'forward': fc,
        'antisymmetry': ac,
        'all': all_conds,
        'a': a,
        'betas': betas,
        'symbols': syms,
    }


# ---------------------------------------------------------------------------
# Groebner-basis interface
# ---------------------------------------------------------------------------

def groebner_basis(conditions, symbols=None, order='grevlex'):
    """
    Thin wrapper around :func:`sympy.groebner`.

    :param conditions: List of polynomial equations (= 0).
    :param symbols: Variable ordering.  Inferred from *conditions* if *None*.
    :param order: Monomial ordering (default ``'grevlex'``).
    :returns: SymPy GroebnerBasis object.
    """
    if not conditions:
        return []
    if symbols is None:
        symbols = _extract_symbols(conditions)
    return sympy.groebner(conditions, *symbols, order=order)


def mathematica_export(conditions) -> str:
    """Export conditions as Mathematica ``Solve`` input."""
    eqs = [sympy.mathematica_code(c) + " == 0" for c in conditions]
    return "Solve[{" + ", ".join(eqs) + "}]"


# ---------------------------------------------------------------------------
# Solution verification
# ---------------------------------------------------------------------------

def verify_conditions(conditions, substitutions) -> tuple[bool, int, sympy.Expr]:
    """
    Verify all conditions are satisfied under *substitutions*.

    :returns: ``(ok, failing_index, residual)``.
    """
    for i, cond in enumerate(conditions):
        val = sympy.simplify(cond.subs(substitutions))
        if val != 0:
            return False, i, val
    return True, -1, sympy.Integer(0)


# ---------------------------------------------------------------------------
# EES character verification
# ---------------------------------------------------------------------------

def verify_ees_character(phi, order: int, tol: float = 1e-10) -> bool:
    """
    Check whether a character satisfies the EES (odd) condition up to *order*.

    Verifies :math:`(\\bar\\phi \\cdot \\phi)(\\tau) = \\varepsilon(\\tau)` for
    all ordered trees up to *order*, where
    :math:`\\bar\\phi(\\tau) = (-1)^{|\\tau|}\\,\\phi(\\tau)`.

    :param phi: A callable mapping planar trees to scalars.
    :param order: Maximum tree order to check.
    :param tol: Numerical tolerance for floating-point residuals.
    :returns: ``True`` if the condition holds for all trees up to *order*.
    """
    from .gentrees import planar_trees_up_to_order
    from .trees import validate_order

    validate_order(order, allow_zero=False)

    def _phi_tree(t):
        return sympy.expand(sympy.sympify(phi(t)))

    def _phi_sign_tree(t):
        return sympy.expand(sign_factor(t) * sympy.sympify(phi(t)))

    # Basis-aware wrappers: shuffle-symmetric on forests
    def _phi(x):
        return _shuffle_sym_value(x, _phi_tree)

    def _phi_sign(x):
        return _shuffle_sym_value(x, _phi_sign_tree)

    for tree in planar_trees_up_to_order(order):
        if tree.list_repr is None:
            continue
        residual = sympy.simplify(
            sympy.expand(_sym_coproduct_eval(tree, _phi_sign, _phi))
        )
        if residual == 0:
            continue
        try:
            if abs(complex(residual)) < tol:
                continue
        except (TypeError, ValueError):
            pass
        return False
    return True
