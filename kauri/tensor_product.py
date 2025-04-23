from dataclasses import dataclass
from typing import Union
from collections import Counter

@dataclass(frozen=True)
class TensorSum():
    term_list: Union[tuple, list, None] #(c, f1, f2)

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
            r += repr(x[0]) + " " + repr(x[1]) + " \u2297 " + repr(x[2])
            r += "+"
        r += repr(self.term_list[-1][0]) + " " + repr(self.term_list[-1][1]) + " \u2297 " + repr(self.term_list[-1][2])
        return r

    # def reduce(self):
    #     #TODO
    #     new_term_list = []
    #
    #     for c, f1, f2 in self.term_list:
    #         f1_reduced = f1.reduce()
    #         f2_reduced = f2.reduce()
    #
    #         for i, (c_, f1_, f2_) in enumerate(new_term_list):
    #             if f1_reduced._equals(f1_) and f2_reduced._equals(f2_):
    #                 old_term_ = new_term_list[i]
    #                 new_term_list[i] = (old_term_[0] + c, old_term_[1], old_term_[2])
    #                 break
    #         else:
    #             new_term_list.append((c, f1_reduced, f2_reduced))
    #
    #     result = tuple(term for term in new_term_list if term[0] != 0)
    #     return TensorSum(result)
    #
    # def __eq__(self, other):
    #     #TODO
    #     if not isinstance(other, TensorSum):
    #         raise
    #     self_reduced = self.reduce()
    #     other_reduced = other.reduce()
    #     return Counter(self_reduced.term_list) == Counter(other_reduced.term_list)

    def __hash__(self):
        return hash(frozenset(Counter(self.term_list).items()))

    def __add__(self, other):
        return TensorSum(self.term_list + other.term_list)

    def __neg__(self):
        return TensorSum(tuple((-x[0], x[1]) for x in self.term_list))

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return TensorSum(tuple((other * x[0], x[1]) for x in self.term_list))
        else:
            raise

    def __iter__(self):
        for f1, f2 in self.term_list:
            yield f1, f2

    def __len__(self):
        return len(self.term_list)

    def __getitem__(self, i):
        return self.term_list[i]
