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
from functools import lru_cache
from math import factorial

from .rk import RK, _check_planar_order, _check_planar_antisymmetric_order
from .maps import Map
from ._protocols import ForestLike
from .trees import EMPTY_PLANAR_TREE, PlanarTree
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

        Builds the same MKW basis-aware character as :meth:`lb_character`,
        but with exact symbolic elementary weights.  Forest values of
        convolution results are therefore evaluated through the MKW forest
        coproduct, not by reusing the base ``prod/k!`` extension.

        :rtype: Map
        """
        if self._symbolic_lb_character is not None:
            return self._symbolic_lb_character

        import sympy
        from .generic_algebra import mkw_base_char_func
        from .mkw.mkw import _as_basis_aware_map
        from .rk import _elementary_symbolic

        a_sym = sympy.Matrix(
            self.s, self.s,
            lambda i, j: sympy.nsimplify(self.a[i][j], rational=True),
        )

        exp_maps = []
        for l in range(self.J):
            b_l = sympy.Matrix(
                1, self.s,
                lambda _, i, l=l: sympy.nsimplify(
                    self.betas[l][i], rational=True),
            )
            cache: dict = {}

            def tree_fn(t, b_l=b_l, cache=cache):
                key = t.list_repr
                if key not in cache:
                    cache[key] = sympy.expand(
                        _elementary_symbolic(key, a_sym, b_l, self.s))
                return cache[key]

            exp_maps.append(_as_basis_aware_map(
                mkw_base_char_func(tree_fn)))

        result = exp_maps[0]
        for l in range(1, self.J):
            result = mkw_map_product(exp_maps[l], result)
        self._symbolic_lb_character = result
        return result

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

        from .mkw.mkw import _as_basis_aware_map

        alpha = self.lb_character()
        sign_alpha = _as_basis_aware_map(
            lambda x: sign_factor(x) * alpha(x))

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


class ReusedStageCFMethod:
    """Low-storage reused-stage CF method.

    This class models the explicit low-storage coefficients as the
    commutator-free row scheme

    ``g_r = exp(sum_k alpha^k_{r,J_r} F_k) ... exp(sum_k alpha^k_{r,1} F_k)(p)``,
    ``F_r = h F_{g_r}``,
    ``y_1 = exp(sum_k beta^k_{J} F_k) ... exp(sum_k beta^k_1 F_k)(p)``,

    where the low-storage ``A_i, B_i`` recurrence only describes how
    exponential prefixes are reused in an implementation.  Each stage row
    starts from the step base point ``p``; it is not a single accumulating
    stage state.  The returned LB character is basis-aware for MKW:
    on an ordered forest ``omega`` it evaluates the row B-series
    coefficient ``g_final(B_+(omega))``.
    """

    def __init__(self, a, b, name=None):
        if not b:
            raise ValueError("At least one exponential required")
        if len(a) != len(b) - 1:
            raise ValueError(
                "Low-storage reused-stage coefficients must satisfy len(a) = len(b) - 1"
            )
        self.A = list(a)
        self.B = list(b)
        self.s = len(self.B)
        self.name = name
        self._lb_character = None
        self._symbolic_lb_character = None
        self._symmetry_defect = None

    @staticmethod
    def _b_plus(trees: tuple) -> PlanarTree:
        return PlanarTree(tuple(t.list_repr for t in trees) + (0,))

    def _owren_rows(self, a_coeffs, b_coeffs):
        """Rows of exponentials induced by the reusable low-storage prefixes."""
        zero = b_coeffs[0] * 0
        one = zero + 1

        rows = [[] for _ in range(self.s + 1)]
        prefix = []
        stage = [zero for _ in range(self.s)]
        stage[0] = one

        for i, coeff in enumerate(b_coeffs):
            prefix.append([coeff * weight for weight in stage])
            if i + 1 < self.s:
                rows[i + 1] = list(prefix)
                stage = [a_coeffs[i] * weight for weight in stage]
                stage[i + 1] = stage[i + 1] + one

        rows[self.s] = list(prefix)
        return rows, zero, one

    def _build_lb_character(self, a_coeffs, b_coeffs) -> Map:
        from .mkw.mkw import _as_basis_aware_map

        rows, zero, one = self._owren_rows(a_coeffs, b_coeffs)
        final_row = self.s

        @lru_cache(maxsize=None)
        def g(row_index: int, exp_count: int, tree_repr):
            if tree_repr is None:
                return one

            children = tuple(PlanarTree(rep) for rep in tree_repr[:-1])
            if not children:
                return one
            if exp_count == 0:
                return zero

            total = zero
            for split in range(len(children) + 1):
                left = self._b_plus(children[:split]).list_repr
                right = self._b_plus(children[split:]).list_repr
                total = total + (
                    g(row_index, exp_count - 1, left)
                    * exp_character(row_index, exp_count, right)
                )
            return total

        @lru_cache(maxsize=None)
        def exp_character(row_index: int, exp_count: int, tree_repr):
            if tree_repr is None:
                return one

            children = tuple(PlanarTree(rep) for rep in tree_repr[:-1])
            if not children:
                return one

            total = one
            for child in children:
                total = total * vector_field(row_index, exp_count, child.list_repr)
            return total / factorial(len(children))

        @lru_cache(maxsize=None)
        def vector_field(row_index: int, exp_count: int, tree_repr):
            coeffs = rows[row_index][exp_count - 1]
            total = zero
            for stage_index, coeff in enumerate(coeffs):
                total = total + coeff * g(
                    stage_index,
                    len(rows[stage_index]),
                    tree_repr,
                )
            return total

        def _char(x):
            if isinstance(x, ForestLike):
                trees = tuple(t for t in x.tree_list if t.list_repr is not None)
                if not trees:
                    return one
                grafted = self._b_plus(trees)
                return g(final_row, len(rows[final_row]), grafted.list_repr)

            if x == EMPTY_PLANAR_TREE:
                return one
            grafted = self._b_plus((x,))
            return g(final_row, len(rows[final_row]), grafted.list_repr)

        return _as_basis_aware_map(_char)

    def projected_rk(self) -> RK:
        """Projected RK tableau induced by the low-storage recurrence."""
        zero = self.B[0] * 0
        one = zero + 1

        rows = [[zero for _ in range(self.s)] for _ in range(self.s)]
        cumulative = [zero for _ in range(self.s)]
        stage = [zero for _ in range(self.s)]
        stage[0] = one

        for i, coeff in enumerate(self.B):
            cumulative = [c + coeff * w for c, w in zip(cumulative, stage)]
            if i + 1 < self.s:
                rows[i + 1] = list(cumulative)
                stage = [self.A[i] * w for w in stage]
                stage[i + 1] = stage[i + 1] + one

        return RK(
            rows,
            cumulative,
            name=(self.name + " (projected)") if self.name else None,
        )

    def lb_character(self) -> Map:
        """Numerical LB character of the reused-stage method."""
        if self._lb_character is None:
            self._lb_character = self._build_lb_character(self.A, self.B)
        return self._lb_character

    def symbolic_lb_character(self) -> Map:
        """Symbolic LB character with exact SymPy coefficients."""
        if self._symbolic_lb_character is not None:
            return self._symbolic_lb_character

        import sympy

        a_sym = [sympy.nsimplify(x, rational=True) for x in self.A]
        b_sym = [sympy.nsimplify(x, rational=True) for x in self.B]
        self._symbolic_lb_character = self._build_lb_character(a_sym, b_sym)
        return self._symbolic_lb_character

    def symmetry_defect_map(self) -> Map:
        """MKW/LB symmetry defect ``D = (sign . alpha) *_MKW alpha``."""
        if self._symmetry_defect is None:
            from .mkw.mkw import _as_basis_aware_map, map_product as mkw_map_product

            alpha = self.lb_character()
            sign_alpha = _as_basis_aware_map(lambda x: sign_factor(x) * alpha(x))
            self._symmetry_defect = mkw_map_product(sign_alpha, alpha)
        return self._symmetry_defect

    def mkw_composition_symmetry_defect_map(self) -> Map:
        """Compatibility alias for the MKW/LB symmetry defect."""
        return self.symmetry_defect_map()

    def planar_order(self, tol: float = 1e-10, limit: int = 10) -> int:
        return _check_planar_order(self.lb_character(), tol, limit)

    def planar_antisymmetric_order(self, tol: float = 1e-10, limit: int = 10) -> int:
        return _check_planar_antisymmetric_order(
            self.symmetry_defect_map(), tol, limit
        )
