import unittest
from kauri import *
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

class LabelledTests(unittest.TestCase):

    def test_init(self):
        t = Tree([[],[]])
        self.assertEqual(t.list_repr, ((0,), (0,),0) )
        t = Tree([[1],0])
        self.assertEqual(t.list_repr, ((1,),0) )

        with self.assertRaises(TypeError):
            Tree(0)
        with self.assertRaises(ValueError):
            t = Tree([[],[],'s'])
        with self.assertRaises(ValueError):
            t = Tree([0, [],[1]])
        with self.assertRaises(ValueError):
            t = Tree([0,0])

    def test_repr(self):
        self.assertEqual(repr(T([[[1],2], [1],2])), '[[[1], 2], [1], 2]')
        self.assertEqual(repr(T([[[1],2], [1],2]).as_forest()), '[[[1], 2], [1], 2]')
        self.assertEqual(repr(T([[[1], 2], [1], 2]).as_forest_sum()), ' 1 * [[[1], 2], [1], 2]')

    def test_equality(self):
        t1 = Tree([[[3], [2], 1], [2]])
        t2 = Tree([[2], [[2], [3], 1]])
        self.assertEqual(t1, t2)

    def test_colors(self):
        self.assertEqual(0, T(None).colors())
        self.assertEqual(1, T([]).colors())
        self.assertEqual(1, T([0]).colors())
        self.assertEqual(10, T([9]).colors())
        self.assertEqual(6, T([[[5],2],1]).colors())