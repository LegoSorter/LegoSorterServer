import logging
import os
import argparse
import time
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

from tensorflow.python.keras.callbacks import Callback
from wandb.keras import WandbCallback
# from models.models import *
# from lego_sorter_server.analysis.classification.models.inceptionClear import InceptionClear
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple
from lego_sorter_server.connection.KaskServerConnector import KaskServerConnector

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import wandb

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 16

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join("lego_sorter_server", "analysis", "classification", "models", "saved"))

CLASSES = ['11211', '15573', '2420', '2431', '2432', '2445', '2456', '2877', '2921', '2926', '3001', '3002', '3003',
           '3004', '3005', '3006', '3007', '3008', '3009', '3010', '30136', '30145', '30157', '3020', '3021', '3022',
           '3023', '3028', '3031', '3032', '3034', '3035', '3037', '30374', '30388', '3039', '30395', '30396', '3040a',
           '30414', '3065', '3066', '3298', '33291', '3460', '3622', '3623', '3659', '3660', '3665a', '3666', '3679',
           '3680', '3710', '3795', '3823', '3832', '4083', '4162', '4175', '4286', '4287a', '43722', '43723', '45677',
           '4600', '4624', '48336', '50950', '54200', '55981', '56891', '60212', '60581', '60592', '60598', '60599',
           '60601', '60608', '60616a', '60623', '6091', '6191', '6215', '6564', '6565', '6636', '84954', '85984',
           '87087', '87414', '87544', '87580', '87697', '88292', '92593', '93273', '93606']


class EvaluateAfterEpoch(Callback):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        acc = self.classifier.eval("test_real_slawek", prefix="hist_", log=True, epoch=epoch)
        if acc > self.best_epoch:
            wandb.run.summary["best_on_test_real_slawek"] = acc
            self.best_epoch = acc
        self.classifier.eval("test_real_bsledz", prefix="hist_", log=True, epoch=epoch, split=0.75)


class TFLegoClassifier(LegoClassifier):
    def __init__(self, classes=None, dataSet=None):
        if not classes:
            classes = CLASSES
        self.dataSet = dataSet
        self.classes = classes
        self.model = None

    def prepare_model(self, model_cls, img_size, weights=None, tf_layers=None, optimizer=None):
        self.model = model_cls.prepare_model(len(self.classes), img_size, weights, tf_layers=tf_layers,
                                             optimizer=optimizer)

    def load_trained_model(self, model_path=DEFAULT_MODEL_PATH):
        if not Path(model_path).exists():
            logging.error(f"[TFLegoClassifier] No model found in {str(model_path)}")
            raise RuntimeError(f"[TFLegoClassifier] No model found in {str(model_path)}")

        self.model = tf.keras.models.load_model(model_path)

    def train(self, epochs, steps_per_epoch):
        if self.model is None:
            self.load_trained_model()
        train_generator = self.dataSet.get_data_generator_train("train")
        validation_generator = self.dataSet.get_data_generator("val", split=0.8)

        filepath = os.path.join("checkpoints", "saved-model-{epoch:02d}-stage1.hdf5")

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                              mode='min')
        return self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[checkpoint_callback, reduce_lr_callback, WandbCallback(), EvaluateAfterEpoch(self)]
        )

    def eval(self, input="test", log=True, prefix="", epoch=None, split=0):
        if self.model is None:
            self.load_trained_model()
        gen = self.dataSet.get_data_generator(input, 1, split)
        loss, accuracy = self.model.evaluate(gen, steps=len(gen)/6)
        if log:
            data = {F'{prefix}{input}: evaluate accuracy': accuracy}
            if epoch:
                data["epoch"] = epoch
            wandb.log(data)
        return accuracy

    def save_model(self, path):
        self.model.save(path)

    def predict_single(self, image: Image.Image) -> ClassificationResults:
        return self.predict([image])

    def predict(self, images: [Image.Image]) -> ClassificationResults:
        if not images:
            return ClassificationResults.empty()
        images_arr = []

        start_time = time.time()
        for im in images:
            transformed = Simple.transform(im)
            img_arr = tf.keras.preprocessing.image.img_to_array(transformed)

            images_arr.append(img_arr)
        gen = ImageDataGenerator(rescale=1. / 255).flow(np.array(images_arr), batch_size=16)

        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        if self.model is None:
            self.load_trained_model()

        predictions = self.model.predict(gen)

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[TFLegoClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        indices = [int(np.argmax(values)) for values in predictions]
        classes = [CLASSES[index] for index in indices]
        scores = [prediction[index] for index, prediction in zip(indices, predictions)]

        return ClassificationResults(classes, scores)


class DataSet:
    def __init__(self, path, batch_size, img_size, w_conf):
        self.path = path
        self.batch_size = batch_size
        self.img_size = img_size
        self.w_conf = w_conf

    def get_data_generator_train(self, type, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        return ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=self.w_conf["rotation_range"],
                                  width_shift_range=self.w_conf["width_shift_range"],
                                  height_shift_range=self.w_conf["height_shift_range"],
                                  validation_split=self.w_conf["validation_split"],
                                  zoom_range=self.w_conf["zoom_range"],
                                  horizontal_flip=True).flow_from_directory(
            os.path.join(self.path, type),
            target_size=self.img_size,
            shuffle=True,
            color_mode="rgb",
            subset="training",
            batch_size=batch_size,
            class_mode='categorical')

    def get_data_generator(self, type, batch_size=None, split=0.0):
        if not batch_size:
            batch_size = self.batch_size
        return ImageDataGenerator(rescale=1. / 255, validation_split=split).flow_from_directory(
            os.path.join(self.path, type),
            target_size=self.img_size,
            shuffle=True,
            subset="training",
            color_mode="rgb",
            batch_size=batch_size,
            class_mode='categorical')


def parse_args():
    ap = argparse.ArgumentParser(description="Train classification")
    ap.add_argument("--dataset", "-i", default="images/dataset")
    ap.add_argument("--epochs", "-e", default=150, type=int)
    ap.add_argument("--tf_layers", "-t", default=150, type=int)
    ap.add_argument("--img_size", "-s", default=75, type=int)
    ap.add_argument("--rotation_range", default=60, type=int)
    ap.add_argument("--zoom_range", default=0, type=float)
    ap.add_argument("--height_shift_range", default=0, type=float)
    ap.add_argument("--shift_range", default=0, type=float)
    ap.add_argument("--validation_split", default=0, type=float)
    ap.add_argument("--steps_per_epoch", default=None, type=int)
    ap.add_argument("--optimizer", "-o", default="adam", choices={"adam", "sgd"})
    ap.add_argument("--model", "-m", default="EfficientNetB0",
                    choices={"EfficientNetB0", "InceptionResNetV2", "InceptionV3", "ResNet50V2", "EfficientNetB5", "VGG16"})

    return ap.parse_args()


def main(args):
    dataset_id = args.dataset.split("/")[-2],
    w_conf = {
        "architecture": args.model,
        "dataset_id": dataset_id[0],
        "dataset": args.dataset,
        "epochs": args.epochs,
        "tf_layers": args.tf_layers,
        "img_size": args.img_size,
        "optimizer": args.optimizer,
        "model": args.model,
        "rotation_range": args.rotation_range,
        "zoom_range": args.zoom_range,
        "height_shift_range": args.shift_range,
        "width_shift_range": args.shift_range,
        "shift_range": args.shift_range,
        "validation_split": args.validation_split,
        "steps_per_epoch": args.steps_per_epoch
    }
    wandb.init(config=w_conf,
               name=F"ds: {dataset_id[0]} s: {args.img_size}px m: {args.model}, epok: {args.epochs}, w. TF: {args.tf_layers}",
               project="praca_magisterska")
    DATASET_PATH = os.path.abspath(os.path.join(args.dataset))

    dataSet = DataSet(DATASET_PATH, BATCH_SIZE, (args.img_size, args.img_size), w_conf)
    network = TFLegoClassifier(dataSet=dataSet)
    # network.prepare_model(Inception)
    modelCls = locate(F"lego_sorter_server.analysis.classification.models.models.{args.model}model")

    network.prepare_model(modelCls, args.img_size, tf_layers=args.tf_layers, optimizer=args.optimizer)
    network.model.summary()

    network.train(args.epochs, args.steps_per_epoch)
    # network.eval("test_renders", prefix="end_", log=True)
    # network.eval("test_real_slawek", prefix="end_", log=True)
    # network.eval("test_real_bsledz", prefix="end_", log=True)
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
