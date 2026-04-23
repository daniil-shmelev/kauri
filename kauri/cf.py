"""
Commutator-free (CF) methods for Lie group integration.

A CF method with *s* stages and *J* exponentials per step is specified by:

- An *s* x *s* coefficient matrix *A* (typically strictly lower triangular for explicit methods).
- *J* weight vectors beta_1, ..., beta_J, each of length *s*.

Update rule (applied right-to-left on the manifold)::

    y_{n+1} = exp(h * sum_i beta_{J,i} F_i) ... exp(h * sum_i beta_{1,i} F_i) y_n

The LB-series character is the MKW convolution::

    alpha = alpha_J *_MKW ... *_MKW alpha_1

where alpha_l is the elementary-weight character of the RK method ``(A,
beta_l)``.  Base exponential characters extend to ordered forests via
the shuffle-symmetric ``1/k!`` rule; convolution results extend via the
paper's forest coproduct (see ``kauri.mkw.forest_coproduct_impl``), and
the combined construction gives associative composition on the MKW Hopf
algebra — the correct Lie-group LB-series character of the method.
"""
from .rk import RK, _check_planar_order, _check_planar_antisymmetric_order
from .maps import Map
from .generic_algebra import sign_factor
from .mkw.mkw import map_product as mkw_map_product


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
        if len(a) != self.s or any(len(row) != self.s for row in a):
            raise ValueError(f"Coefficient matrix A must be {self.s}x{self.s}, matching beta vector length")
        for l, beta in enumerate(betas):
            if len(beta) != self.s:
                raise ValueError(f"betas[{l}] has length {len(beta)}, expected {self.s}")
        self.a = a
        self.betas = betas
        self.name = name
        self.b = [sum(betas[l][i] for l in range(self.J)) for i in range(self.s)]
        self._lb_character = None
        self._symbolic_lb_character = None
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

        Computed as ``alpha_J *_MKW ... *_MKW alpha_1`` with each
        ``alpha_l`` wrapped as a shuffle-symmetric base character on the
        MKW Hopf algebra (the returned :class:`Map` carries
        ``extension="shuffle"``).  This is the correct LB-series character
        for a Lie-group integrator: the tree values are the RK elementary
        weights of the individual exponentials, and composition via the
        MKW convolution captures the non-abelian Lie-group flow.

        The result is cached on first call.

        :rtype: Map
        """
        if self._lb_character is not None:
            return self._lb_character

        from .generic_algebra import mkw_base_char_func
        from .mkw.mkw import _as_basis_aware_map

        exp_maps = [
            _as_basis_aware_map(
                mkw_base_char_func(
                    self.exponential_rk(l).elementary_weights_map().func))
            for l in range(self.J)
        ]
        result = exp_maps[0]
        for l in range(1, self.J):
            result = mkw_map_product(exp_maps[l], result)
        self._lb_character = result
        return result

    def symbolic_lb_character(self) -> Map:
        """
        Symbolic LB-series character: same algebra as :meth:`lb_character`
        but each tree is mapped to an exact :class:`sympy.Expr` (typically
        a :class:`sympy.Rational`) instead of a float.

        Internally delegates to
        :func:`kauri.manifold_ees.symbolic_lb_character` after converting
        this method's concrete tableau entries to :class:`sympy.Rational`
        via :func:`sympy.nsimplify`.

        :rtype: Map
        """
        if self._symbolic_lb_character is not None:
            return self._symbolic_lb_character

        import sympy
        from .manifold_ees import symbolic_lb_character as _sym_char

        a_sym = sympy.Matrix(
            self.s, self.s,
            lambda i, j: sympy.nsimplify(self.a[i][j], rational=True),
        )
        betas_sym = [
            [sympy.nsimplify(self.betas[l][i], rational=True)
             for i in range(self.s)]
            for l in range(self.J)
        ]
        cache: dict = {}

        def _char(t):
            key = t.list_repr
            if key not in cache:
                cache[key] = _sym_char(t, a_sym, betas_sym, self.s, self.J)
            return cache[key]

        m = Map(_char)
        self._symbolic_lb_character = m
        return m

    def symmetry_defect_map(self) -> Map:
        """
        Symmetry defect ``D = (sign . alpha) *_MKW alpha``.

        ``D(tau) = epsilon(tau)`` for all ``|tau| <= q`` iff the CF method
        has planar antisymmetric order >= *q*.

        The result is cached on first call.

        :rtype: Map
        """
        if self._symmetry_defect is not None:
            return self._symmetry_defect

        from .generic_algebra import mkw_base_char_func
        from .mkw.mkw import _as_basis_aware_map

        alpha = self.lb_character()
        # sign · alpha as a shuffle-symmetric base character: on trees
        # it's (-1)^|tau|·alpha(tau); the forest extension follows the
        # canonical 1/k! rule via mkw_base_char_func.
        sign_alpha = _as_basis_aware_map(
            mkw_base_char_func(lambda t: sign_factor(t) * alpha(t)))

        self._symmetry_defect = mkw_map_product(sign_alpha, alpha)
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
