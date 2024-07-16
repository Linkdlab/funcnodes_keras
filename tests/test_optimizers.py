import funcnodes as fn
import unittest
from tensorflow.keras.optimizers import Optimizer
from funcnodes_keras.optimizers import (
    _SGD,
    _RMSprop,
    _Adam,
    _AdamW,
    _Adadelta,
    _Adagrad,
    _Adamax,
    _Adafactor,
    _Nadam,
    _Ftrl,
    _Lion,
)


class TestSGDNodes(unittest.IsolatedAsyncioTestCase):
    async def test_sgd_default(self):
        optimizer_node: fn.Node = _SGD()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)

class TestRMSpropNodes(unittest.IsolatedAsyncioTestCase):
    async def test_rmsprop_default(self):
        optimizer_node: fn.Node = _RMSprop()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
        
class TestAdamNodes(unittest.IsolatedAsyncioTestCase):
    async def test_adam_default(self):
        optimizer_node: fn.Node = _Adam()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
class TestAdamWNodes(unittest.IsolatedAsyncioTestCase):
    async def test_adamw_default(self):
        optimizer_node: fn.Node = _AdamW()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
        
class TestAdadeltaNodes(unittest.IsolatedAsyncioTestCase):
    async def test_adadelta_default(self):
        optimizer_node: fn.Node = _Adadelta()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
class TestAdagradNodes(unittest.IsolatedAsyncioTestCase):
    async def test_adagrad_default(self):
        optimizer_node: fn.Node = _Adagrad()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)

class TestAdamaxNodes(unittest.IsolatedAsyncioTestCase):
    async def test_adamax_default(self):
        optimizer_node: fn.Node = _Adamax()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
class TestAdafactorNodes(unittest.IsolatedAsyncioTestCase):
    async def test_adafactor_default(self):
        optimizer_node: fn.Node = _Adafactor()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
class TestNadamNodes(unittest.IsolatedAsyncioTestCase):
    async def test_nadam_default(self):
        optimizer_node: fn.Node = _Nadam()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)
        
class TestFtrlNodes(unittest.IsolatedAsyncioTestCase):
    async def test_ftrl_default(self):
        optimizer_node: fn.Node = _Ftrl()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)

class TestLionNodes(unittest.IsolatedAsyncioTestCase):
    async def test_lion_default(self):
        optimizer_node: fn.Node = _Lion()
        self.assertIsInstance(optimizer_node, fn.Node)
        await optimizer_node
        self.assertIsInstance(optimizer_node.outputs["out"].value, Optimizer)