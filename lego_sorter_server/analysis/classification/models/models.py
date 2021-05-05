import tensorflow as tf

# create the base pre-trained model
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3, Xception, EfficientNetB0
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.applications.efficientnet import EfficientNetB3, EfficientNetB5
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2


class InceptionResNetV2model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        base_model = InceptionResNetV2(weights=weights or 'imagenet', include_top=False, pooling='max',
                                       classes=cls_count,
                                       input_shape=(img_size, img_size, 3))
        if tf_layers:
            for layer in base_model.layers[:-tf_layers]:
                layer.trainable = False
        x = base_model.output

        predictions = Dense(cls_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model


class InceptionV3model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        base_model = InceptionV3(weights=weights or 'imagenet', include_top=False, pooling='max',
                                 classes=cls_count,
                                 input_shape=(img_size, img_size, 3))
        if tf_layers:
            for layer in base_model.layers[:-int(tf_layers)]:
                layer.trainable = False
        x = base_model.output

        predictions = Dense(cls_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model


class ResNet50V2model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        base_model = ResNet50V2(weights=weights or 'imagenet', include_top=False, pooling='max',
                                classes=cls_count,
                                input_shape=(img_size, img_size, 3))
        if tf_layers:
            for layer in base_model.layers[:-int(tf_layers)]:
                layer.trainable = False
        x = base_model.output

        predictions = Dense(cls_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model


class EfficientNetB5model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        base_model = EfficientNetB5(weights=weights or 'imagenet', include_top=False, pooling='max',
                                    classes=cls_count,
                                    input_shape=(img_size, img_size, 3))
        if tf_layers:
            for layer in base_model.layers[:-int(tf_layers)]:
                layer.trainable = False
        x = base_model.output

        predictions = Dense(cls_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model


class VGG16model():
    @staticmethod
    def prepare_model(cls_count, img_size, weights=None, tf_layers=None, optimizer=None):
        base_model = VGG16(weights=weights or 'imagenet', include_top=False, pooling='max',
                           classes=cls_count,
                           input_shape=(img_size, img_size, 3))
        if tf_layers:
            for layer in base_model.layers[:-int(tf_layers)]:
                layer.trainable = False
        x = base_model.output

        predictions = Dense(cls_count, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        return model
