from .trees import *

# These are just convenience functions to be used with "apply" and "apply_product"

def ident(t):
    return t

def counit(t):
    if t == Tree(None):
        return 1
    else:
        return 0

def S(t):
    return t.antipode()

def exact_weights(t):
    return 1. / t.factorial()

def sqrt(t):
    return t.id_sqrt()

def odd_component(t):
    return t.minus()

def even_component(t):
    return t.plus()

def RK_internal_weights(i, t, A, b, s):
    return sum(A[i][j] * RK_derivative_weights(j, t, A, b, s) for j in range(s))

def RK_derivative_weights(i, t, A, b, s):
    if t == Tree(None) or t == Tree([]):
        return 1
    else:
        out = 1
        for subtree in t.unjoin().treeList:
            out *= RK_internal_weights(i, subtree, A, b, s)
        return out


def RK_elementary_weights(t, A, b):
    s = len(b)
    return sum(b[i] * RK_derivative_weights(i, t, A, b, s) for i in range(s))