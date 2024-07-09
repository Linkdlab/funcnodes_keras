import funcnodes as fn
import unittest
import numpy as np
from funcnodes_keras.datasets import (
    _mnist,
    _cifar10,
    _cifar100,
    _imdb,
    _reuters,
    _fashion_mnist,
    _california_housing,
    VersionMode,
)


class TestDatasetsNodes(unittest.IsolatedAsyncioTestCase):
    async def test_mnist(self):
        ds: fn.Node = _mnist()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(x_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))

    async def test_cifar10(self):
        ds: fn.Node = _cifar10()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 1))
        self.assertEqual(y_test.shape, (10000, 1))

    async def test_cifar100(self):
        ds: fn.Node = _cifar100()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 1))
        self.assertEqual(y_test.shape, (10000, 1))

    async def test_imdb(self):
        ds: fn.Node = _imdb()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(x_train[0], list)
        self.assertIsInstance(x_train[0][0], int)

    async def test_reuters(self):
        ds: fn.Node = _reuters()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(x_train[0], list)
        self.assertIsInstance(x_train[0][0], int)

    async def test_fashion_mnist(self):
        ds: fn.Node = _fashion_mnist()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertEqual(x_train.shape, (60000, 28, 28))
        self.assertEqual(x_test.shape, (10000, 28, 28))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(y_test.shape, (10000,))

    async def test_california_housing(self):
        ds: fn.Node = _california_housing()
        version = VersionMode.large
        if version == VersionMode.small:
            num_samples = 480
        elif version == VersionMode.large:
            num_samples = 16512
        ds.inputs["version"].value = version
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        y_train = ds.outputs["y_train"].value
        x_test = ds.outputs["x_test"].value
        y_test = ds.outputs["y_test"].value

        self.assertEqual(x_train.shape, (num_samples, 8))
        self.assertEqual(x_test.shape, (int(num_samples / 4), 8))
        self.assertEqual(y_train.shape, (num_samples,))
        self.assertEqual(y_test.shape, (int(num_samples / 4),))
