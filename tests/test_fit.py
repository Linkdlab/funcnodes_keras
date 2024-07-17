import funcnodes as fn
import unittest
import numpy as np
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import KerasTensor

from funcnodes_keras.datasets import _mnist
from funcnodes_keras.metrics import _SparseCategoricalAccuracy
from funcnodes_keras.losses import _SparseCategoricalCrossentropy
from funcnodes_keras.optimizers import _RMSprop

from funcnodes_keras.applications import _ResNet50
from funcnodes_keras.layers import _Input, _Dense, Activation
from funcnodes_keras.models import _Model
from funcnodes_keras.fit import (
    _fit,
    _compile,
    _predict,
)

random_image_array = np.random.randint(0, 256, size=(224, 224, 3))


class TestKerasExampleNodes(unittest.IsolatedAsyncioTestCase):
    async def test_dataset_default(self):
        ds: fn.Node = _mnist()
        self.assertIsInstance(ds, fn.Node)
        await ds
        x_train = ds.outputs["x_train"].value
        x_test = ds.outputs["x_test"].value
        y_train = ds.outputs["y_train"].value
        y_test = ds.outputs["y_test"].value
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(x_test, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        # Preprocess the data (these are NumPy arrays)
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255

        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")

        # Reserve 10,000 samples for validation
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        input_layer: fn.Node = _Input()
        input_layer.inputs["shape_1_height_or_sequence_length"].value = 784
        input_layer.inputs["shape_2_width_or_input_dim"].value = 1

        self.assertIsInstance(input_layer, fn.Node)

        dense_layer_1: fn.Node = _Dense()
        dense_layer_1.inputs["input_model"].connect(input_layer.outputs["out"])
        dense_layer_1.inputs["units"].value = 64
        dense_layer_1.inputs["activation"].value = Activation.relu
        self.assertIsInstance(dense_layer_1, fn.Node)

        dense_layer_2: fn.Node = _Dense()
        dense_layer_2.inputs["input_model"].connect(dense_layer_1.outputs["out"])
        dense_layer_2.inputs["units"].value = 64
        dense_layer_2.inputs["activation"].value = Activation.relu
        self.assertIsInstance(dense_layer_2, fn.Node)

        dense_layer_3: fn.Node = _Dense()
        dense_layer_3.inputs["input_model"].connect(dense_layer_2.outputs["out"])
        dense_layer_3.inputs["units"].value = 10
        dense_layer_3.inputs["activation"].value = Activation.softmax
        self.assertIsInstance(dense_layer_3, fn.Node)

        prepared_model: fn.Node = _Model()
        prepared_model.inputs["input"].connect(input_layer.outputs["out"])
        prepared_model.inputs["output"].connect(dense_layer_3.outputs["out"])
        self.assertIsInstance(prepared_model, fn.Node)

        loss: fn.Node = _SparseCategoricalCrossentropy()
        self.assertIsInstance(loss, fn.Node)

        metric: fn.Node = _SparseCategoricalAccuracy()
        self.assertIsInstance(metric, fn.Node)

        optimizer: fn.Node = _RMSprop()
        self.assertIsInstance(optimizer, fn.Node)

        compile: fn.Node = _compile()
        compile.inputs["model"].connect(prepared_model.outputs["out"])
        compile.inputs["loss"].connect(loss.outputs["out"])
        compile.inputs["optimizer"].connect(optimizer.outputs["out"])
        compile.inputs["metrics"].connect(metric.outputs["out"])
        self.assertIsInstance(compile, fn.Node)
        fit: fn.Node = _fit()
        fit.inputs["model"].connect(compile.outputs["compiled_model"])
        fit.inputs["x"].value = x_train
        fit.inputs["y"].value = y_train
        fit.inputs["x_val"].value = x_val
        fit.inputs["y_val"].value = y_val
        self.assertIsInstance(fit, fn.Node)

        await fn.run_until_complete(
            fit,
            compile,
            metric,
            loss,
            optimizer,
            prepared_model,
            dense_layer_3,
            dense_layer_2,
            dense_layer_1,
            input_layer,
        )
        fitted_model = fit.outputs["fitted_model"].value
        metrics_dictionary = fit.outputs["metrics_dictionary"].value
        history = fit.outputs["history"].value
        self.assertIsInstance(history, History)
        self.assertIsInstance(metrics_dictionary, dict)
        self.assertIsInstance(fitted_model, Model)


class TestPredictingNodes(unittest.IsolatedAsyncioTestCase):
    async def test_predict_random_pretrained(self):
        ft_model: fn.Node = _ResNet50()
        ft_model.inputs["include_top"].value = False
        self.assertIsInstance(ft_model, fn.Node)

        t_model: fn.Node = _predict()
        t_model.inputs["model"].connect(ft_model.outputs["out"])
        t_model.inputs["x"].value = np.expand_dims(random_image_array, axis=0)
        self.assertIsInstance(t_model, fn.Node)
        await fn.run_until_complete(t_model, ft_model)
        prediction = t_model.outputs["out"].value
        print(prediction)
