import os
import sys
import unittest
import funcnodes_keras as fnmodule
from funcnodes_keras import fit
from funcnodes import lib
import test_applications, test_losses, test_fit, test_metrics, test_optimizers, test_utilities  # noqa E402


sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)  # in case test folder is not in sys path
from all_nodes_test_base import TestAllNodesBase  # noqa: E402

sub_test_classes = []

for mod in (
    test_applications,
    test_losses,
    test_fit,
    test_metrics,
    test_optimizers,
    test_utilities,
):
    for cls in mod.__dict__.values():
        if isinstance(cls, type) and issubclass(cls, unittest.IsolatedAsyncioTestCase):
            sub_test_classes.append(cls)


class TestAllNodes(TestAllNodesBase):
    # in this test class all nodes should be triggered at least once to mark them as testing
    sub_test_classes = sub_test_classes

    ignore_nodes = (
        lib.flatten_shelf(fnmodule.DATASETS_NODE_SHELFE)[0]
        + lib.flatten_shelf(fnmodule.LAYERS_NODE_SHELFE)[0]
        + lib.flatten_shelf(fnmodule.MODELS_NODE_SHELFE)[0]
        + [
            fit._evaluate,
            fit._train_on_batch,
            fit._test_on_batch,
            fit._predict_on_batch,
        ]
    )
