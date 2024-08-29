import funcnodes as fn
import unittest
from funcnodes_keras import utilities


class TestUtilities(unittest.IsolatedAsyncioTestCase):
    async def test_to_categorical(self):
        utility_node: fn.Node = utilities._to_categorical()
        utility_node.inputs["x"].value = [0, 1, 2, 3]
        utility_node.inputs["num_classes"].value = 4
        self.assertIsInstance(utility_node, fn.Node)
        await utility_node
        self.assertEqual(utility_node.outputs["out"].value.shape, (4, 4))
