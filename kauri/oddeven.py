# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

#TODO: docs
from .trees import Tree
from .bck import antipode
from .generic_algebra import _apply
from .maps import Map, ident, sign
from functools import cache

@cache
def _id_sqrt(self): #Id^{1/2}
    if self.equals(Tree(None)):
        return Tree(None) * 1
    if self.equals(Tree([])):
        return Tree([]) * 0.5
    else:
        out = (ident ** 2)(self) - 2 * self
        out = _apply(out, _id_sqrt)
        out = (self - out) * 0.5
        out = out.simplify()
        return out

id_sqrt = Map(_id_sqrt)
minus = ((sign & antipode) * ident) & id_sqrt
plus = ident * (minus & antipode)