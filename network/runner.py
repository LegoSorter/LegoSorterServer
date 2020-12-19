import os

import tensorflow as tf
# inceptionv3 imports
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# callback imports
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd

from models.inception import Inception

BATCH_SIZE = 64
EPOCHS = 10
IMG_SIZE = (224, 224)
DATASET_PATH = R"C:\LEGO_repo\DataAlchemist\DataSet"


class Network:
    def __init__(self, model_cls, dataSet, weights=None):
        self.dataSet = dataSet
        self.model = model_cls.prepare_model(dataSet.length(), weights)

    def train(self, epochs):
        train_generator = self.dataSet.get_data_generator("train")
        validation_generator = self.dataSet.get_data_generator("val")

        filepath = os.path.join("checkpoints", "saved-model-{epoch:02d}-stage1.hdf5")

        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False,
                                              mode='max')
        return self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint_callback]
        )

    def eval(self):
        self.model.evaluate(self.dataSet.get_data_generator("test"), steps=20)


class DataSet:
    def __init__(self, path, batch_size, img_size):
        self.path = path
        self.batch_size = batch_size
        self.img_size = img_size
        self.classes = os.listdir(os.path.join(path, "train"))

    def length(self):
        return len(self.classes)

    def get_data_generator(self, type):
        return ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            os.path.join(self.path, type),
            target_size=self.img_size,
            shuffle=True,
            batch_size=self.batch_size,
            class_mode='categorical')

def main():
    dataSet = DataSet(DATASET_PATH, BATCH_SIZE, IMG_SIZE)
    network = Network(Inception, dataSet)
    network.train(EPOCHS)
    network.eval()


main()
