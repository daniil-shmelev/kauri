from trees import *

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