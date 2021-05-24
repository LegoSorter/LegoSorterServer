import tensorflow as tf

# create the base pre-trained model
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3, Xception, EfficientNetB0
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.applications.efficientnet import EfficientNetB3, EfficientNetB5
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential


class InceptionResNetV2model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        img_augmentation = Sequential(
            [
                preprocessing.Rescaling(scale=1. / 255)

            ],
            name="img_augmentation",
        )

        x = img_augmentation(inputs)
        base_model = InceptionResNetV2(weights=weights or 'imagenet', include_top=False, input_tensor=x)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model


class InceptionV3model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        img_augmentation = Sequential(
            [
                preprocessing.Rescaling(scale=1. / 255)

            ],
            name="img_augmentation",
        )

        x = img_augmentation(inputs)
        base_model = InceptionV3(weights=weights or 'imagenet', include_top=False, input_tensor=x)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model


class ResNet50V2model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        img_augmentation = Sequential(
            [
                preprocessing.Rescaling(scale=1. / 255)

            ],
            name="img_augmentation",
        )

        x = img_augmentation(inputs)
        base_model = ResNet50V2(weights=weights or 'imagenet', include_top=False, input_tensor=x)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model


class EfficientNetB0model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        base_model = EfficientNetB0(weights=weights or 'imagenet', include_top=False, input_tensor=inputs)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

class EfficientNetB3model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        base_model = EfficientNetB3(weights=weights or 'imagenet', include_top=False, input_tensor=inputs)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model

class EfficientNetB5model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        base_model = EfficientNetB5(weights=weights or 'imagenet', include_top=False, input_tensor=inputs)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model


class VGG16model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        inputs = layers.Input(shape=(img_size, img_size, 3))
        img_augmentation = Sequential(
            [
                preprocessing.Rescaling(scale=1. / 255)

            ],
            name="img_augmentation",
        )

        x = img_augmentation(inputs)
        base_model = VGG16(weights=weights or 'imagenet', include_top=False, input_tensor=x)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = layers.BatchNormalization()(x)

        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False

        top_dropout_rate = 0.3
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(cls_count, activation="softmax", name="pred")(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                      metrics=["accuracy"])
        return model
