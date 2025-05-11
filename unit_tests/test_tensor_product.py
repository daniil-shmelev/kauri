import unittest
from kauri import TensorProductSum
from kauri import Tree as T

trees = [T(None),
         T([]),
         T([[]]),
         T([[],[]]),
         T([[[]]]),
         T([[],[],[]]),
         T([[],[[]]]),
         T([[[],[]]]),
         T([[[[]]]])]

class TensorProductSumTests(unittest.TestCase):
    def test_tensor(self):
        t1 = T([]) @ T([[]]) + T([]) * T([]) @ T([[],[]])
        t2 = TensorProductSum([(1, T([]), T([[]])), (1, T([]) * T([]), T([[],[]]))])
        self.assertEqual(t1, t2)