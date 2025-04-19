from dataclasses import dataclass
from typing import Union
from collections import Counter

@dataclass(frozen=True)
class TensorSum():
    term_list: Union[tuple, list, None]

    def __post_init__(self):
        tuple_list = tuple(tuple(x) for x in self.term_list)
        object.__setattr__(self, 'term_list', tuple_list)

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        memodict[id(self)] = self
        return self

    def __repr__(self):
        if self.term_list is None:
            return "0"
        r = ""
        for x in self.term_list[:-1]:
            r += repr(x[0]) + "" + repr(x[1])
            r += "+"
        r += repr(self.term_list[-1][0]) + "" + repr(self.term_list[-1][1])
        return r

    def __hash__(self):
        return hash(frozenset(Counter(self.term_list).items()))

    def __add__(self, other):
        return TensorSum(self.term_list + other.term_list)

    def __neg__(self):
        return TensorSum(tuple(tuple(-x[0], x[1]) for x in self.term_list))

    def __sub__(self, other):
        self + (-other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return TensorSum(tuple(tuple(other * x[0], x[1]) for x in self.term_list))
        else:
            raise

    def __iter__(self):
        for subtree, branches in self.term_list:
            yield subtree, branches

    def __len__(self):
        return len(self.term_list)

    def __getitem__(self, i):
        return self.term_list[i]
