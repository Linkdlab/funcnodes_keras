import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from all_nodes_test_base import TestAllNodesBase
import funcnodes as fn
import funcnodes_keras as fnmodule
import dataclasses
from . import (
    test_applications,
    test_datasets,
    test_losses,
    test_fit,
    test_metrics,
    test_optimizers,
)
import unittest

sub_test_classes = []

for mod in (
    test_applications,
    test_datasets,
    test_losses,
    test_fit,
    test_metrics,
    test_optimizers,
):
    for cls in test_applications.__dict__.values():
        if isinstance(cls, type) and issubclass(cls, unittest.IsolatedAsyncioTestCase):
            sub_test_classes.append(cls)


class TestAllNodes(TestAllNodesBase):
    ### in this test class all nodes should be triggered at least once to mark them as testing
    sub_test_classes = sub_test_classes

    def test_all_nodes(self):
        pass
