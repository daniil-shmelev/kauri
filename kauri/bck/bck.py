"""
Front-end for the BCK module
"""
from ..bck_impl import _counit, _coproduct, _antipode
from ..maps import Map

counit = Map(_counit)
counit.__doc__ = """
The counit :math:`\\varepsilon_{BCK}` of the BCK Hopf algebra.

:type: Map

Example usage::
    
    import kauri as kr
    import kauri.bck as bck

    bck.counit(kr.Tree(None)) # Returns 1
    bck.counit(kr.Tree([])) # Returns 0
"""

antipode = Map(_antipode)
antipode.__doc__ = """
The antipode :math:`S_{BCK}` of the BCK Hopf algebra.

:type: Map

Example usage::

    import kauri as kr
    import kauri.bck as bck

    t = kr.Tree([[[]],[]])
    bck.antipode(t)
"""

coproduct = Map(_coproduct)
coproduct.__doc__ = """
The coproduct :math:`\\Delta_{BCK}` of the BCK Hopf algebra.

:type: Map

Example usage::

    import kauri as kr
    import kauri.bck as bck

    bck.coproduct(kr.Tree([])) # Returns 1 ∅ ⊗ []+1 [] ⊗ ∅
    bck.coproduct(kr.Tree([[]])) # Returns 1 [[]] ⊗ ∅+1 ∅ ⊗ [[]]+1 [] ⊗ []
"""

def map_product(f, g):
    """
    Returns the product of maps in the BCK Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

    :type f: Map
    :type g: Map
    :rtype: Map

    .. note::
        `bck.map_product(f,g)` is equivalent to the Map operator `f * g`

    Example usage::

        import kauri as kr
        import kauri.bck as bck

        ident = kr.Map(lambda x : x)
        counit = bck.map_product(ident, bck.antipode) # Equivalent to indent * bck.antipode
    """
    return f * g

def map_power(f, exponent):
    """
    Returns the power of a map in the BCK Hopf algebra, where the product of functions is defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

    and negative powers are defined as :math:`f^{-n} = f^n \\circ S_{BCK}`,
    where :math:`S_{BCK}` is the BCK antipode.

    :type f: Map
    :param exponent: Exponent
    :type exponent: int
    :rtype: Map

    .. note::
        `bck.map_power(f, n)` is equivalent to the Map operator `f ** n`

    Example usage::

        import kauri as kr
        import kauri.bck as bck

        ident = kr.Map(lambda x : x)
        S = bck.map_power(ident, -1) # antipode, equivalent to ident ** (-1)
        ident_sq = bck.map_power(ident, 2) # identity squared, equivalent to ident ** 2
    """
    return f ** exponent
