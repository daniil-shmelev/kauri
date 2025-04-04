from .trees import Tree, Forest, ForestSum

# These are just convenience functions to be used with "apply" and "apply_product"

def ident(t):
    """The identity map, :math:`t \mapsto t`"""
    return t

def counit(t):
    """The counit map, :math:`t \mapsto \epsilon(t)`, where :math:`\epsilon(t) = 1` if :math:`t = \emptyset` and :math:`0` otherwise."""
    if t == Tree(None):
        return 1
    else:
        return 0

def S(t):
    """The antipode map, :math:`t \mapsto S(t)`"""
    return t.antipode()

def exact_weights(t):
    """The map :math:`t \mapsto 1 / t!`, giving the coefficients of the B-series of the exact solution to the ODE
    :math:`dy/dt = f(y)`."""
    return 1. / t.factorial()

def _RK_internal_weights(i, t, A, b, s):
    return sum(A[i][j] * _RK_derivative_weights(j, t, A, b, s) for j in range(s))

def _RK_derivative_weights(i, t, A, b, s):
    if t == Tree(None) or t == Tree([]):
        return 1
    else:
        out = 1
        for subtree in t.unjoin().tree_list:
            out *= _RK_internal_weights(i, subtree, A, b, s)
        return out


def RK_elementary_weights(t, A, b):
    """The elementary weights function for a Runge-Kutta scheme with parameters, :math:`(A,b)`."""
    if t == Tree(None):
        return 1
    s = len(b)
    return sum(b[i] * _RK_derivative_weights(i, t, A, b, s) for i in range(s))