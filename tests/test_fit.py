import funcnodes as fn
import unittest

from typing import Literal, Union, Optional, Iterator, Tuple, Callable, List
import numpy as np
from funcnodes import Shelf, NodeDecorator
from exposedfunctionality import controlled_wrapper
from enum import Enum
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop

from funcnodes_keras.applications import (
    _ResNet50
)

from funcnodes_keras.fit import (
    _fit,
    _compile,
    _evaluate,
    _predict,
    _predict_on_batch,
    _test_on_batch,
    _train_on_batch,
)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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


inputs = Input(shape=(784,), name="digits")
x = Dense(64, activation="relu", name="dense_1")(inputs)
x = Dense(64, activation="relu", name="dense_2")(x)
outputs = Dense(10, activation="softmax", name="predictions")(x)

prepared_model = Model(inputs=inputs, outputs=outputs)


class TestCompilingingFittingNodes(unittest.IsolatedAsyncioTestCase):
    async def test_compile_keras_example(self):
        ft_model: fn.Node = _compile()
        ft_model.inputs["model"].value = prepared_model
        ft_model.inputs["loss"].value = SparseCategoricalCrossentropy()
        ft_model.inputs["metrics"].value = [SparseCategoricalAccuracy()]
        self.assertIsInstance(ft_model, fn.Node)

        t_model: fn.Node = _fit()
        t_model.inputs["model"].connect(ft_model.outputs["compiled_model"])
        t_model.inputs["x"].value = x_train
        t_model.inputs["y"].value = y_train
        t_model.inputs["x_val"].value = x_val
        t_model.inputs["y_val"].value = y_val
        self.assertIsInstance(t_model, fn.Node)
        
        
        await fn.run_until_complete(t_model, ft_model)
        fitted_model = t_model.outputs["fitted_model"].value
        metrics_dictionary = t_model.outputs["metrics_dictionary"].value
        history = t_model.outputs["history"].value
        self.assertIsInstance(history, History)
        self.assertIsInstance(metrics_dictionary, dict)
        self.assertIsInstance(fitted_model, Model)


random_image_array = np.random.randint(0, 256, size=(224, 224, 3))

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