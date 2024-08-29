import funcnodes as fn
import unittest
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import (
    Accuracy,
    BinaryAccuracy,
    CategoricalAccuracy,
    SparseCategoricalAccuracy,
    TopKCategoricalAccuracy,
    SparseTopKCategoricalAccuracy,
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    KLDivergence,
    Poisson,
    MeanSquaredError,
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredLogarithmicError,
    CosineSimilarity,
    LogCoshError,
    R2Score,
    AUC,
    Precision,
    Recall,
    TruePositives,
    TrueNegatives,
    FalsePositives,
    FalseNegatives,
    PrecisionAtRecall,
    RecallAtPrecision,
    SensitivityAtSpecificity,
    SpecificityAtSensitivity,
    F1Score,
    FBetaScore,
    IoU,
    BinaryIoU,
    OneHotIoU,
    OneHotMeanIoU,
    MeanIoU,
    Hinge,
    SquaredHinge,
    CategoricalHinge,
)
from funcnodes_keras import metrics


class TestMetricsNodes(unittest.IsolatedAsyncioTestCase):
    async def test_all_metrics(self):
        for m in [
            Accuracy,
            BinaryAccuracy,
            CategoricalAccuracy,
            SparseCategoricalAccuracy,
            TopKCategoricalAccuracy,
            SparseTopKCategoricalAccuracy,
            BinaryCrossentropy,
            CategoricalCrossentropy,
            SparseCategoricalCrossentropy,
            KLDivergence,
            Poisson,
            MeanSquaredError,
            RootMeanSquaredError,
            MeanAbsoluteError,
            MeanAbsolutePercentageError,
            MeanSquaredLogarithmicError,
            CosineSimilarity,
            LogCoshError,
            R2Score,
            AUC,
            Precision,
            Recall,
            TruePositives,
            TrueNegatives,
            FalsePositives,
            FalseNegatives,
            F1Score,
            FBetaScore,
            Hinge,
            SquaredHinge,
            CategoricalHinge,
        ]:
            self.assertTrue(
                hasattr(metrics, f"_{m.__name__}"), msg=f"Missing metric: {m.__name__}"
            )

            nodeclass = getattr(metrics, f"_{m.__name__}")
            self.assertTrue(
                issubclass(nodeclass, fn.Node), msg=f"Missing metric: {m.__name__}"
            )

            metric_node: fn.Node = nodeclass()
            self.assertIsInstance(metric_node, fn.Node)
            await metric_node
            self.assertIsInstance(metric_node.outputs["out"].value, Metric, msg=m)
            self.assertIsInstance(metric_node.outputs["out"].value, m, msg=m)

    async def test_precision_at_recall(self):
        metric_node = metrics._PrecisionAtRecall()
        metric_node.inputs["recall"].value = 0.5
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, PrecisionAtRecall)

    async def test_recall_at_precision(self):
        metric_node = metrics._RecallAtPrecision()
        metric_node.inputs["precision"].value = 0.5
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, RecallAtPrecision)

    async def test_sensitivity_at_specificity(self):
        metric_node = metrics._SensitivityAtSpecificity()
        metric_node.inputs["specificity"].value = 0.5
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(
            metric_node.outputs["out"].value, SensitivityAtSpecificity
        )

    async def test_specificity_at_sensitivity(self):
        metric_node = metrics._SpecificityAtSensitivity()
        metric_node.inputs["sensitivity"].value = 0.5
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(
            metric_node.outputs["out"].value, SpecificityAtSensitivity
        )

    async def test_iou(self):
        metric_node = metrics._IoU()
        metric_node.inputs["num_classes"].value = 2
        metric_node.inputs["target_class_ids"].value = [0, 1]
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, IoU)

    async def test_binary_iou(self):
        metric_node = metrics._BinaryIoU()
        metric_node.inputs["target_class_ids"].value = [0, 1]
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, BinaryIoU)

    async def test_onehot_iou(self):
        metric_node = metrics._OneHotIoU()
        metric_node.inputs["num_classes"].value = 2
        metric_node.inputs["target_class_ids"].value = [0, 1]
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, OneHotIoU)

    async def test_onehot_mean_iou(self):
        metric_node = metrics._OneHotMeanIoU()
        metric_node.inputs["num_classes"].value = 2
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, OneHotMeanIoU)

    async def test_mean_iou(self):
        metric_node = metrics._MeanIoU()
        metric_node.inputs["num_classes"].value = 2
        self.assertIsInstance(metric_node, fn.Node)
        await metric_node
        self.assertIsInstance(metric_node.outputs["out"].value, Metric)
        self.assertIsInstance(metric_node.outputs["out"].value, MeanIoU)
