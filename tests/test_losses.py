import funcnodes as fn
import unittest
from tensorflow.keras.losses import Loss
from funcnodes_keras.losses import (
    _BinaryCrossentropy,
    _BinaryFocalCrossentropy,
    _CategoricalCrossentropy,
    _CategoricalFocalCrossentropy,
    _SparseCategoricalCrossentropy,
    _Poisson,
    _KLDivergence,
    _CTC,
    _MeanSquaredError,
    _MeanAbsoluteError,
    _MeanAbsolutePercentageError,
    _MeanSquaredLogarithmicError,
    _CosineSimilarity,
    _Huber,
    _LogCosh,
    _Hinge,
    _SquaredHinge,
    _CategoricalHinge,
)


class TestBinaryCrossentropyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_binary_crossentropy_default(self):
        loss_node: fn.Node = _BinaryCrossentropy()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestBinaryFocalCrossentropyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_binary_focal_crossentropy_default(self):
        loss_node: fn.Node = _BinaryFocalCrossentropy()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestCategoricalCrossentropyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_categorical_crossentropy_default(self):
        loss_node: fn.Node = _CategoricalCrossentropy()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestCategoricalFocalCrossentropyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_categorical_focal_crossentropy_default(self):
        loss_node: fn.Node = _CategoricalFocalCrossentropy()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestSparseCategoricalCrossentropyNodes(unittest.IsolatedAsyncioTestCase):
    async def test_categorical_crossentropy_default(self):
        loss_node: fn.Node = _SparseCategoricalCrossentropy()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestPoissonNodes(unittest.IsolatedAsyncioTestCase):
    async def test_poisson_default(self):
        loss_node: fn.Node = _Poisson()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestKLDivergenceNodes(unittest.IsolatedAsyncioTestCase):
    async def test_kldivergence_default(self):
        loss_node: fn.Node = _KLDivergence()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestCTCNodes(unittest.IsolatedAsyncioTestCase):
    async def test_ctc_default(self):
        loss_node: fn.Node = _CTC()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestMeanSquaredErrorNodes(unittest.IsolatedAsyncioTestCase):
    async def test_mean_squared_error_default(self):
        loss_node: fn.Node = _MeanSquaredError()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestMeanAbsoluteErrorNodes(unittest.IsolatedAsyncioTestCase):
    async def test_mean_absolute_error_default(self):
        loss_node: fn.Node = _MeanAbsoluteError()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestMeanAbsolutePercentageErrorNodes(unittest.IsolatedAsyncioTestCase):
    async def test_mean_absolute_percentage_error_default(self):
        loss_node: fn.Node = _MeanAbsolutePercentageError()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestMeanSquaredLogarithmicErrorNodes(unittest.IsolatedAsyncioTestCase):
    async def test_mean_squared_logarithmic_error_default(self):
        loss_node: fn.Node = _MeanSquaredLogarithmicError()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestCosineSimilarityNodes(unittest.IsolatedAsyncioTestCase):
    async def test_cosine_similarity_default(self):
        loss_node: fn.Node = _CosineSimilarity()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestHuberNodes(unittest.IsolatedAsyncioTestCase):
    async def test_huber_default(self):
        loss_node: fn.Node = _Huber()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestLogCoshNodes(unittest.IsolatedAsyncioTestCase):
    async def test_log_cosh_default(self):
        loss_node: fn.Node = _LogCosh()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestHingeNodes(unittest.IsolatedAsyncioTestCase):
    async def test_hinge_default(self):
        loss_node: fn.Node = _Hinge()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestSquaredHingeNodes(unittest.IsolatedAsyncioTestCase):
    async def test_squared_hinge_default(self):
        loss_node: fn.Node = _SquaredHinge()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)


class TestCategoricalHingeNodes(unittest.IsolatedAsyncioTestCase):
    async def test_categorical_hinge_default(self):
        loss_node: fn.Node = _CategoricalHinge()
        self.assertIsInstance(loss_node, fn.Node)
        await loss_node
        self.assertIsInstance(loss_node.outputs["out"].value, Loss)
