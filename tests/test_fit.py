import funcnodes as fn
import unittest
import numpy as np
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from funcnodes_keras.datasets import _mnist
from funcnodes_keras.metrics import _SparseCategoricalAccuracy
from funcnodes_keras.losses import _SparseCategoricalCrossentropy
from funcnodes_keras.optimizers import _RMSprop

from funcnodes_keras.applications import _ResNet50

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

        # TODO: add funcnode base model
        inputs = Input(shape=(784,), name="digits")
        x = Dense(64, activation="relu", name="dense_1")(inputs)
        x = Dense(64, activation="relu", name="dense_2")(x)
        outputs = Dense(10, activation="softmax", name="predictions")(x)

        prepared_model = Model(inputs=inputs, outputs=outputs)

        self.assertIsInstance(prepared_model, Model)
        loss: fn.Node = _SparseCategoricalCrossentropy()
        self.assertIsInstance(loss, fn.Node)

        metric: fn.Node = _SparseCategoricalAccuracy()
        self.assertIsInstance(metric, fn.Node)

        optimizer: fn.Node = _RMSprop()
        self.assertIsInstance(optimizer, fn.Node)

        # await fn.run_until_complete(metric, loss, optimizer)
        # self.assertIsInstance(loss.outputs["out"].value, Loss)
        # self.assertIsInstance(metric.outputs["out"].value, Metric)
        # self.assertIsInstance(optimizer.outputs["out"].value, Optimizer)

        compile: fn.Node = _compile()
        compile.inputs["model"].value = prepared_model
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

        await fn.run_until_complete(fit, compile, metric, loss, optimizer)
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
