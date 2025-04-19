from kauri.bck_impl import _counit, _coproduct, _antipode
from ..maps import Map

counit = Map(_counit)
antipode = Map(_antipode)
coproduct = Map(_coproduct)

def map_product(map1, map2):
    return  map1 * map2

def map_power(map, exponent):
    return map ** exponent