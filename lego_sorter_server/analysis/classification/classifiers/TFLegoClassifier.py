import logging
import os
import argparse
from datetime import datetime
from pydoc import locate

import numpy as np
import tensorflow as tf
# inceptionv3 imports
from PIL import Image
# callback imports
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# from models.models import *
# from lego_sorter_server.analysis.classification.models.inceptionClear import InceptionClear
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple
from lego_sorter_server.connection.KaskServerConnector import KaskServerConnector

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 32
IMG_SIZE = (299, 299)
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join("lego_sorter_server", "analysis", "classification", "models", "saved"))

CLASSES = [
    "3001",
    "3002",
    "3003",
    "3004",
    "3710",
    "3009",
    "3010",
    "3007",
    "3034",
    "3832"
]


class TFLegoClassifier:
    def __init__(self, classes=None, dataSet=None):
        if not classes:
            classes = CLASSES
        self.dataSet = dataSet
        self.classes = classes
        self.model = None

    def prepare_model(self, model_cls, weights=None):
        self.model = model_cls.prepare_model(len(self.classes), weights)

    def load_trained_model(self, model_path=DEFAULT_MODEL_PATH):
        if not Path(model_path).exists():
            logging.error(f"[TFLegoClassifier] No model found in {str(model_path)}")
            raise RuntimeError(f"[TFLegoClassifier] No model found in {str(model_path)}")

        self.model = tf.keras.models.load_model(model_path)

    def train(self, epochs, tb_name):
        if self.model is None:
            self.load_trained_model()
        train_generator = self.dataSet.get_data_generator("train")
        validation_generator = self.dataSet.get_data_generator("val")

        filepath = os.path.join("checkpoints", "saved-model-{epoch:02d}-stage1.hdf5")

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                              mode='min')
        tb_callback = tf.keras.callbacks.TensorBoard(tb_name, update_freq=1)
        return self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint_callback, reduce_lr_callback, tb_callback]
        )

    def eval(self, input="test"):
        if self.model is None:
            self.load_trained_model()
        self.model.evaluate(self.dataSet.get_data_generator(input, 1), steps=20)

    def save_model(self, path):
        self.model.save(path)

    def classify(self, image: Image.Image) -> ClassificationResults:
        return self.predict_from_pil([image])

    def predict_from_pil(self, images: [Image.Image]) -> ClassificationResults:
        if not images:
            return ClassificationResults.empty()
        images_arr = []

        for im in images:
            transformed = Simple.transform(im)
            img_arr = tf.keras.preprocessing.image.img_to_array(transformed)

            images_arr.append(img_arr)
        gen = ImageDataGenerator(rescale=1. / 255).flow(np.array(images_arr), batch_size=1)

        if self.model is None:
            self.load_trained_model()

        predictions = self.model.predict(gen)
        indices = [int(np.argmax(values)) for values in predictions]
        classes = [CLASSES[index] for index in indices]
        scores = [prediction[index] for index, prediction in zip(indices, predictions)]

        return ClassificationResults(classes, scores)


class DataSet:
    def __init__(self, path, batch_size, img_size):
        self.path = path
        self.batch_size = batch_size
        self.img_size = img_size

    def get_data_generator(self, type, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return ImageDataGenerator(rescale=1. / 255, rotation_range=15, horizontal_flip=True,
                                  vertical_flip=True).flow_from_directory(
            os.path.join(self.path, type),
            target_size=self.img_size,
            shuffle=True,
            batch_size=batch_size,
            class_mode='categorical')


def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Train classification")
    ap.add_argument("--input", "-i", default="images/dataset",
                    help="path to a tsv file or folder with tsv files")
    ap.add_argument("--epochs", "-e", default=10)
    ap.add_argument("--model", "-m", default="InceptionClear",
                    choices={"Inception", "InceptionClear", "VGG16", "Xception"})

    return ap.parse_args()


def main(args):
    DATASET_PATH = os.path.abspath(os.path.join(args.input))
    dataSet = DataSet(DATASET_PATH, BATCH_SIZE, IMG_SIZE)
    network = TFLegoClassifier(dataSet=dataSet)
    # network.prepare_model(Inception)
    modelCls = locate(F"lego_sorter_server.analysis.classification.models.models.{args.model}")
    network.prepare_model(modelCls)
    network.model.summary()
    date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    network.train(args.epochs, os.path.join("boards", F"{args.model}_{args.epochs}_{date}"))
    network.eval()
    network.eval("test_renders")
    # im = Image.open(R"C:\LEGO_repo\LegoSorterServer\images\storage\stored\3003\ccRZ_3003_1608582857061.jpg")
    # print(network.predict_from_pil([im]))
    network.save_model(DEFAULT_MODEL_PATH)

    # network.load_trained_model()
    # # im = Image.open(R"C:\LEGO_repo\LegoSorterServer\images\storage\stored\3001\oymy_3001_1608586741032.jpg")
    # im = Image.open(R"C:\LEGO_repo\LegoSorterServer\images\storage\stored\3003\ccRZ_3003_1608582857061.jpg")
    # print(network.predict_from_pil([im]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
