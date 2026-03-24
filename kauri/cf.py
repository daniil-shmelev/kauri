"""
Commutator-free (CF) methods for Lie group integration.

A CF method with *s* stages and *J* exponentials per step is specified by:

- An explicit *s* x *s* coefficient matrix *A* (strictly lower triangular).
- *J* weight vectors beta_1, ..., beta_J, each of length *s*.

Update rule (applied right-to-left on the manifold)::

    y_{n+1} = exp(h * sum_i beta_{J,i} F_i) ... exp(h * sum_i beta_{1,i} F_i) y_n

The LB-series character is the ordered (planar BCK) convolution::

    alpha = alpha_J *_pbck ... *_pbck alpha_1

where alpha_l is the elementary-weight character of the RK method ``(A, beta_l)``.
BCH corrections are accounted for automatically by the noncommutative convolution.
"""
from .rk import RK, _check_planar_order, _check_planar_antisymmetric_order
from .maps import Map
from .generic_algebra import sign_factor
from .pbck.pbck import map_product as pbck_map_product


class CFMethod:
    """
    A commutator-free Lie group integrator.

    :param a: Explicit s x s coefficient matrix.
    :param betas: List of J weight vectors, each of length s.
                  ``betas[0]`` is the innermost (first-applied) exponential.
    :param name: Optional name for display.
    """

    def __init__(self, a, betas, name=None):
        if not betas:
            raise ValueError("At least one exponential required")
        self.s = len(betas[0])
        self.J = len(betas)
        self.a = a
        self.betas = betas
        self.name = name
        self.b = [sum(betas[l][i] for l in range(self.J)) for i in range(self.s)]
        self._lb_character = None
        self._symmetry_defect = None

    def projected_rk(self) -> RK:
        """The projected RK method with ``b = sum_l beta_l``."""
        return RK(self.a, self.b,
                  name=(self.name + " (projected)") if self.name else None)

    def exponential_rk(self, l: int) -> RK:
        """The RK method ``(A, beta_l)`` for the *l*-th exponential (0-indexed)."""
        return RK(self.a, self.betas[l])

    def lb_character(self) -> Map:
        """
        The LB-series character on ordered trees.

        Computed as ``alpha_J *_pbck ... *_pbck alpha_1``.
        The result is cached on first call.

        :rtype: Map
        """
        if self._lb_character is not None:
            return self._lb_character

        exp_maps = [self.exponential_rk(l).elementary_weights_map()
                    for l in range(self.J)]
        result = exp_maps[0]
        for l in range(1, self.J):
            result = pbck_map_product(exp_maps[l], result)
        self._lb_character = result
        return result

    def symmetry_defect_map(self) -> Map:
        """
        Symmetry defect ``D = (sign . alpha) *_pbck alpha``.

        ``D(tau) = epsilon(tau)`` for all ``|tau| <= q`` iff the CF method
        has planar antisymmetric order >= *q*.

        The result is cached on first call.

        :rtype: Map
        """
        if self._symmetry_defect is not None:
            return self._symmetry_defect

        alpha = self.lb_character()
        sign_alpha = Map(lambda t: sign_factor(t) * alpha(t))
        self._symmetry_defect = pbck_map_product(sign_alpha, alpha)
        return self._symmetry_defect

    def planar_order(self, tol: float = 1e-10, limit: int = 10) -> int:
        """
        Order of the CF method on ordered trees.

        :param tol: Tolerance for evaluating conditions.
        :param limit: Maximum order to check.
        :rtype: int
        """
        return _check_planar_order(self.lb_character(), tol, limit)

    def planar_antisymmetric_order(self, tol: float = 1e-10, limit: int = 10) -> int:
        """
        Planar antisymmetric order of the CF method.

        :param tol: Tolerance for evaluating conditions.
        :param limit: Maximum order to check.
        :rtype: int
        """
        return _check_planar_antisymmetric_order(
            self.symmetry_defect_map(), tol, limit)
