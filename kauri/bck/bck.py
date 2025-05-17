"""
Front-end for the BCK module
"""
from ..bck_impl import _counit, _coproduct, _antipode
from ..maps import Map
from ..trees import Tree
from ..tensor_product import TensorProductSum

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

def coproduct(t : Tree) -> TensorProductSum:
    """
    The coproduct :math:`\\Delta_{BCK}` of the BCK Hopf algebra.

    :param t: tree
    :type t: Tree
    :rtype: TensorProductSum

    Example usage::

        import kauri as kr
        import kauri.bck as bck

        bck.coproduct(kr.Tree([])) # Returns 1 ∅ ⊗ []+1 [] ⊗ ∅
        bck.coproduct(kr.Tree([[]])) # Returns 1 [[]] ⊗ ∅+1 ∅ ⊗ [[]]+1 [] ⊗ []
    """
    if not isinstance(t, Tree):
        raise TypeError("Argument to bck.coproduct must be a Tree, not " + str(type(t)))
    return _coproduct(t)

def map_product(f : Map, g : Map) -> Map:
    """
    Returns the product of maps in the BCK Hopf algebra, defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

    .. note::
        `bck.map_product(f,g)` is equivalent to the Map operator `f * g`

    :param f: f
    :type f: Map
    :param g: g
    :type g: Map
    :rtype: Map

    Example usage::

        import kauri as kr
        import kauri.bck as bck

        ident = kr.Map(lambda x : x)
        counit = bck.map_product(ident, bck.antipode) # Equivalent to indent * bck.antipode
    """
    if not (isinstance(f, Map) and isinstance(g, Map)):
        raise TypeError("Arguments in bck.map_product must be of type Map, not " + str(type(f)) + " and " + str(type(g)))
    return f * g

def map_power(f : Map, exponent : int) -> Map:
    """
    Returns the power of a map in the BCK Hopf algebra, where the product of functions is defined by

    .. math::

        (f \\cdot g)(t) := \\mu \\circ (f \\otimes g) \\circ \\Delta_{BCK} (t)

    and negative powers are defined as :math:`f^{-n} = f^n \\circ S_{BCK}`,
    where :math:`S_{BCK}` is the BCK antipode.

    .. note::
        `bck.map_power(f, n)` is equivalent to the Map operator `f ** n`

    :param f: f
    :type f: Map
    :param exponent: exponent
    :type exponent: int

    Example usage::

        import kauri as kr
        import kauri.bck as bck

        ident = kr.Map(lambda x : x)
        S = bck.map_power(ident, -1) # antipode, equivalent to ident ** (-1)
        ident_sq = bck.map_power(ident, 2) # identity squared, equivalent to ident ** 2
    """
    if not isinstance(f, Map):
        raise TypeError("f must be a Map, not " + str(type(f)))
    if not isinstance(exponent, int):
        raise TypeError("exponent must be an int, not " + str(type(exponent)))
    return f ** exponent
