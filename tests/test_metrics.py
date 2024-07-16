import funcnodes as fn
import unittest
from tensorflow.keras.metrics import Metric
from funcnodes_keras.metrics import _Accuracy,_SparseCategoricalCrossentropy, _R2Score, _AUC



class TestAccuracyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_accuracy_default(self):
        metric_node: fn.Node = _Accuracy()
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        
class TestSparseCategoricalCrossentropyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_sparse_categorical_crossentropy_default(self):
        metric_node: fn.Node = _SparseCategoricalCrossentropy()
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)

class TestR2ScoreNodes(unittest.IsolatedAsyncioTestCase):
    async def test_r2score_default(self):
        metric_node: fn.Node = _R2Score()
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)

class TestAUCNodes(unittest.IsolatedAsyncioTestCase):
    async def test_auc_default(self):
        metric_node: fn.Node = _AUC()
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)