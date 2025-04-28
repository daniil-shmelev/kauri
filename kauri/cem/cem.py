"""
Front-end for the CEM module
"""
from ..cem_impl import _counit, _coproduct, _antipode
from ..maps import Map
from ..trees import Tree, TensorProductSum
from ..generic_algebra import _func_power

counit = Map(_counit)
counit.__doc__ = """
The counit :math:`\\varepsilon_{CEM}` of the CEM Hopf algebra.

:type: Map

Example usage::

    from kauri import Tree
    import kauri.cem as cem

    cem.counit(Tree([])) # Returns 1
    cem.counit(Tree([[]])) # Returns 0
"""

antipode = Map(_antipode)
antipode.__doc__ = """
The antipode :math:`S_{CEM}` of the CEM Hopf algebra.

:type: Map

Example usage::

    from kauri import Tree
    import kauri.cem as cem

    t = Tree([[[]],[]])
    cem.antipode(t)
"""

def coproduct(t : Tree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta_{CEM}` of the CEM Hopf algebra.

    :param t: tree
    :type t: Tree
    :rtype: TensorProductSum

    Example usage::

        from kauri import Tree
        import kauri.cem as cem

        cem.coproduct(Tree([])) # Returns 1 [] ⊗ []
        cem.coproduct(Tree([[]])) # Returns 1 [] ⊗ [[]]+1 [[]] ⊗ []
    """
    return _coproduct(t)

def map_product(f : Map, g : Map) -> Map:
    """
    Returns the product of maps in the CEM Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{CEM} (t)

    .. note::
        `cem.map_product(f,g)` is equivalent to the Map operator `f ^ g`

    Example usage::

        import kauri as kr
        import kauri.cem as cem

        ident = kr.Map(lambda x : x)
        counit = cem.map_product(ident, cem.antipode) # Equivalent to ident ^ cem.antipode
    """
    return  f ^ g

def map_power(f : Map, exponent : int) -> Map:
    """
    Returns the power of a map in the CEM Hopf algebra, where the product of functions is defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{CEM} (t)

    and negative powers are defined as :math:`f^{-n} = f^n \\circ S_{CEM}`,
    where :math:`S_{CEM}` is the CEM antipode.

    Example usage::

        import kauri as kr
        import kauri.cem as cem

        ident = kr.Map(lambda x : x)
        S = cem.map_power(ident, -1) # antipode
        ident_sq = cem.map_power(ident, 2) # identity squared
    """

    if not isinstance(exponent, int):
        raise TypeError("Exponent in Map power must be int, got " + str(type(exponent)) + " instead.")

    return Map(lambda x: _func_power(x, f.func, exponent, _coproduct, _counit, _antipode))
