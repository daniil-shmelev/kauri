from ..cem_impl import _counit, _coproduct, _antipode
from ..maps import Map
from ..generic_algebra import _func_power

counit = Map(_counit)
antipode = Map(_antipode)
coproduct = Map(_coproduct)

def map_product(map1, map2):
    return  map1 ^ map2

def map_power(map, exponent):
    if not isinstance(exponent, int):
        raise ValueError("Map.__pow__ received invalid exponent")

    return Map(lambda x: _func_power(x, map.func, exponent, _coproduct, _counit, _antipode))