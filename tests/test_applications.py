import funcnodes as fn
import unittest

# from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from funcnodes_keras.applications import (
    Pooling,
    Weights,
    ClassifierActivation,
    _Xception,
    _VGG16,
    _VGG19,
    _ResNet50,
    _ResNet50V2,
    _ResNet101,
    _ResNet101V2,
    _ResNet152,
    _ResNet152V2,
    _InceptionV3,
    _InceptionResNetV2,
    _MobileNet,
    _MobileNetV2,
    _MobileNetV3Small,
    _MobileNetV3Large,
    _DenseNet121,
    _DenseNet169,
    _DenseNet201,
    _NASNetLarge,
    _NASNetMobile,
    _EfficientNetB0,
    _EfficientNetB1,
    _EfficientNetB2,
    _EfficientNetB3,
    _EfficientNetB4,
    _EfficientNetB5,
    _EfficientNetB6,
    _EfficientNetB7,
    _EfficientNetV2B0,
    _EfficientNetV2B1,
    _EfficientNetV2B2,
    _EfficientNetV2B3,
    _EfficientNetV2S,
    _EfficientNetV2M,
    _EfficientNetV2L,
    _ConvNeXtTiny,
    _ConvNeXtSmall,
    _ConvNeXtBase,
    _ConvNeXtLarge,
    _ConvNeXtXLarge,
)

input_size = 224


class TestXception(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _Xception()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _Xception()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _Xception()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _Xception()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestVGG16(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _VGG16()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _VGG16()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _VGG16()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _VGG16()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestVGG19(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _VGG19()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _VGG19()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _VGG19()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _VGG19()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestResNet50(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ResNet50()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ResNet50()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ResNet50()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ResNet50()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestResNet50V2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ResNet50V2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ResNet50V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ResNet50V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ResNet50V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestResNet101(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ResNet101()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ResNet101()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ResNet101()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ResNet101()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestResNet101V2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ResNet101V2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ResNet101V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ResNet101V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ResNet101V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestResNet152(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ResNet152()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ResNet152()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ResNet152()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ResNet152()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestResNet152V2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ResNet152V2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ResNet152V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ResNet152V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ResNet152V2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestInceptionV3(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _InceptionV3()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _InceptionV3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _InceptionV3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _InceptionV3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestInceptionResNetV2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _InceptionResNetV2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _InceptionResNetV2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _InceptionResNetV2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _InceptionResNetV2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestMobileNet(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _MobileNet()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _MobileNet()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _MobileNet()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _MobileNet()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestMobileNetV2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _MobileNetV2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _MobileNetV2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _MobileNetV2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _MobileNetV2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestMobileNetV3Small(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _MobileNetV3Small()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _MobileNetV3Small()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _MobileNetV3Small()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _MobileNetV3Small()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestMobileNetV3Large(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _MobileNetV3Large()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _MobileNetV3Large()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _MobileNetV3Large()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _MobileNetV3Large()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestDenseNet121(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _DenseNet121()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _DenseNet121()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _DenseNet121()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _DenseNet121()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestDenseNet169(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _DenseNet169()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _DenseNet169()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _DenseNet169()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _DenseNet169()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestDenseNet201(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _DenseNet201()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _DenseNet201()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _DenseNet201()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _DenseNet201()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestNASNetMobile(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _NASNetMobile()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _NASNetMobile()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _NASNetMobile()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _NASNetMobile()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestNASNetLarge(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _NASNetLarge()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _NASNetLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _NASNetLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _NASNetLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB0(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB0()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB0()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB0()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB0()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB1(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB1()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB1()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB1()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB1()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB3(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB3()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB4(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB4()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB4()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB4()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB4()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB5(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB5()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB5()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB5()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB5()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB6(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB6()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB6()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB6()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB6()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetB7(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetB7()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetB7()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetB7()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetB7()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2B0(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2B0()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2B0()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2B0()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2B0()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2B1(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2B1()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2B1()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2B1()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2B1()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2B2(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2B2()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2B2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2B2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2B2()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2B3(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2B3()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2B3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2B3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2B3()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2S(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2S()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2S()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2S()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2S()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2M(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2M()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2M()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2M()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2M()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestEfficientNetV2L(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _EfficientNetV2L()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _EfficientNetV2L()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _EfficientNetV2L()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _EfficientNetV2L()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestConvNeXtTiny(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ConvNeXtTiny()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ConvNeXtTiny()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ConvNeXtTiny()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ConvNeXtTiny()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestConvNeXtSmall(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ConvNeXtSmall()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ConvNeXtSmall()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ConvNeXtSmall()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ConvNeXtSmall()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestConvNeXtBase(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ConvNeXtBase()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ConvNeXtBase()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ConvNeXtBase()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ConvNeXtBase()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestConvNeXtLarge(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ConvNeXtLarge()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ConvNeXtLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ConvNeXtLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ConvNeXtLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)


class TestConvNeXtXLarge(unittest.IsolatedAsyncioTestCase):
    async def test_default_parameters(self):
        keras_model: fn.Node = _ConvNeXtXLarge()
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_pooling(self):
        keras_model: fn.Node = _ConvNeXtXLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["pooling"].value = Pooling.max
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_weights(self):
        keras_model: fn.Node = _ConvNeXtXLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["weights"].value = Weights.imagenet
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)

    async def test_classifier_activation(self):
        keras_model: fn.Node = _ConvNeXtXLarge()
        keras_model.inputs["include_top"].value = False
        keras_model.inputs["input_shape_height"].value = input_size
        keras_model.inputs["input_shape_width"].value = input_size
        keras_model.inputs["classifier_activation"].value = ClassifierActivation.relu
        self.assertIsInstance(keras_model, fn.Node)
        await keras_model
        out = keras_model.outputs["out"]
        self.assertIsInstance(out.value, Model)
