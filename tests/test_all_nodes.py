import sys
import os
import unittest
import funcnodes_keras as fnmodule

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from all_nodes_test_base import TestAllNodesBase  # noqa E402

from . import (  # noqa E402
    test_applications,
    test_datasets,
    test_losses,
    test_fit,
    test_metrics,
    test_optimizers,
)


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
    # in this test class all nodes should be triggered at least once to mark them as testing
    sub_test_classes = sub_test_classes

    ignore_nodes = fnmodule.DATASETS_NODE_SHELFE["nodes"] + []
