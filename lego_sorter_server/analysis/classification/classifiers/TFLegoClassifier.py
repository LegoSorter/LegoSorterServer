import argparse
import logging
import os
import time
from pathlib import Path
from pydoc import locate

import numpy as np
import tensorflow as tf
import wandb
# inceptionv3 imports
from PIL import Image
# callback imports
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback
from wandb.keras import WandbCallback

import pandas
# from models.models import *
# from lego_sorter_server.analysis.classification.models.inceptionClear import InceptionClear
from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.helpers.aug import get_augmenting_sequence, get_no_augmenting_sequence
from lego_sorter_server.analysis.classification.helpers.generator import DataGenerator
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 4

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join("lego_sorter_server", "analysis", "classification", "models", "saved", "saved-model-40-stage1.hdf5"))

CLASSES = ['10197', '10201', '10314', '10928', '11090', '11153', '11211', '11212', '11213', '11214', '11215', '11272',
           '11458', '11476', '11477', '11478', '120493', '131673', '13349', '13547', '13548', '13731', '14395', '14417',
           '14419', '14704', '14716', '14720', '14769',
           '15068', '15070', '15092', '15100', '15254', '15332', '15379', '15395', '15397', '15460', '15461', '15470',
           '15535', '15573', '15672', '15706', '15712', '158788', '15967', '16577', '17114', '17485', '18649', '18651',
           '18653', '18674', '18838', '18969', '19159',
           '20896', '21229', '216731', '22385', '22388', '22390', '22391', '22885', '22888', '22889', '22890', '22961',
           '2357', '239356', '24014', '24122', '2412b', '2419', '2420', '242434', '24246', '24299', '24316', '24375',
           '2441', '2445', '2450', '24505', '2453b', '2454', '2456', '2460', '2476a', '2486', '24866', '2540', '254579',
           '26047', '2639', '2654', '26601', '26604', '267165', '27255', '27262', '27266', '2730', '2736', '274829',
           '27940', '2853', '2854', '28653', '2877', '2904', '2926', '292629', '296435', '30000', '3001',
           '3002', '3003', '3004', '30044', '30046', '3005', '30069', '3008', '3009', '30099', '3010', '30136', '30157',
           '30165', '3020', '3021', '3022', '30237b', '3024', '3028', '3031', '3032', '3033', '3034', '3035', '30357',
           '30361c', '30363', '30367c', '3037', '3038',
           '30387', '3040b', '30414', '3045', '30503', '30552', '30553', '30565', '3062', '3068', '3069b', '3070b',
           '3185', '32000', '32002', '32013', '32016', '32017', '32028', '32034', '32039', '32054', '32056', '32059',
           '32062', '32064a', '32073', '32123b', '32124',
           '32140', '32184', '32187', '32192', '32198', '32250', '32291', '32316', '32348', '3245', '32526', '32529',
           '32530', '32557', '32828', '32932', '3298', '33291', '33299b', '33909', '3460', '35044', '3622', '3623',
           '3633', '3639', '3640', '3659', '3660', '3665', '3666'
    , '3673', '3676', '3679', '3680', '36840', '36841', '3700', '3701', '3705', '3706', '3707', '3710', '3713',
           '374125', '3747b', '3795', '3832', '3895', '392043', '3941', '3942c', '3957', '3958', '39739', '4032a',
           '40490', '4073', '4081b', '4083', '40902', '413097'
    , '41530', '41532', '4162', '41677', '41678', '41682', '41740', '41747', '41748', '41762', '41768', '41769',
           '41770', '4185', '42003', '42023', '4216', '4218', '42446', '4274', '4282', '4286', '4287b', '43708',
           '43712', '43713', '43898', '44126', '44568', '4460b'
    , '44728', '4477', '44809', '44861', '4488', '4490', '4510', '4519', '45590', '456218', '45677', '4600', '465007',
           '4727', '4733', '47397', '47398', '4740', '4742', '47456', '474589', '47753', '47755', '47905', '48092',
           '48171', '48336', '4865b', '4871', '48723',
           '48729b', '48933', '48989', '496432', '49668', '50304', '50305', '50373', '50950', '51739', '52031',
           '523081', '52501', '53899', '54383', '54384', '55013', '551028', '56596', '569005', '57519', '57520',
           '57585', '57895', '57909b', '58090', '59426', '59443',
           '60032', '6005', '6014', '6020', '60470b', '60471', '60474', '60475b', '60476', '60477', '60478', '60479',
           '60481', '60483', '60484', '60581', '60592', '60593', '60594', '60596', '60598', '60599', '60607', '60608',
           '60616b', '60621', '60623', '608036', '6081', '60897', '6091', '61070', '61071', '6111', '61252', '612598',
           '61409', '614655', '61484', '6157', '6182', '6215', '6222', '6231', '6232', '6233', '62462', '63864',
           '63868', '64225', '64288', '64391', '64393', '64681', '64683', '64712', '6536', '6541', '6553', '6558',
           '6587', '6628', '6632', '6636', '72454', '74261', '77206', '822931', '84954', '85080', '852929', '853045',
           '85984', '87079', '87081', '87083', '87087', '87544', '87580', '87609', '87620', '87697', '88292', '88323',
           '88646', '88930', '901078', '90195', '90202', '90609', '90611', '90630', '915460', '92013', '92092', '92582',
           '92583', '92589', '92907', '92947', '92950', '93273', '93274', '93606', '94161', '959666', '966967', '98100',
           '98197', '98262', '98283', '98560', '99008', '99021', '99206', '99207', '99773', '99780', '99781']


class EvaluateAfterEpoch(Callback):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        pass
        if epoch % 5 == 0:
            acc = self.classifier.eval("test", prefix="hist_", log=True, epoch=epoch, split=0.80)
            if acc > self.best_epoch:
                wandb.run.summary["best_on_test_real"] = acc
                self.best_epoch = acc
        # self.classifier.eval("test_real_bsledz", prefix="hist_", log=True, epoch=epoch, split=0.75)


class TFLegoClassifier(LegoClassifier):
    def __init__(self, classes=None, dataSet=None):
        if not classes:
            classes = CLASSES
        self.dataSet = dataSet
        self.classes = classes
        self.model = None

    def prepare_model(self, model_cls, img_size, weights=None, tf_layers=None):
        self.model = model_cls.prepare_model(len(self.classes), img_size, weights, tf_layers=tf_layers)

    def load_trained_model(self, model_path=DEFAULT_MODEL_PATH):
        if not Path(model_path).exists():
            logging.error(f"[TFLegoClassifier] No model found in {str(model_path)}")
            raise RuntimeError(f"[TFLegoClassifier] No model found in {str(model_path)}")

        self.model = tf.keras.models.load_model(model_path)

    def train(self, epochs, steps_per_epoch):
        if self.model is None:
            self.load_trained_model()
        train_generator = self.dataSet.get_data_generator_train_aug("train")
        print(list(train_generator.extract_labels()))
        exit()
        # datas, labels = train_generator.get_data(0)
        # for data in datas:
        #     Image.fromarray(data, 'RGB').show()

        validation_generator = self.dataSet.get_data_generator_train_aug("val", reduction=0)

        filepath = os.path.join("checkpoints", "saved-model-{epoch:02d}-stage1.hdf5")

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5,
                                                                  verbose=1)
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                              mode='min')
        return self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=min(300, len(validation_generator)),
            callbacks=[checkpoint_callback, reduce_lr_callback, WandbCallback(), EvaluateAfterEpoch(self)]
            # save=model=false
        )

    def eval(self, input="test", log=True, prefix="", epoch=None, split=0):
        if self.model is None:
            self.load_trained_model()
        gen = self.dataSet.get_data_generator(input, 1, split)
        accuracy = self.model.evaluate(gen, steps=len(gen) / 10)[1]
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
            transformed = Simple.transform(im, 300)
            img_arr = tf.keras.preprocessing.image.img_to_array(transformed)

            images_arr.append(img_arr)
        gen = ImageDataGenerator().flow(np.array(images_arr), batch_size=16)

        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        if self.model is None:
            self.load_trained_model()

        predictions = self.model.predict(gen)

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[TFLegoClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        indices = [int(np.argmax(values)) for values in predictions]
        classes = [CLASSES[index] for index in indices]
        # print("PREDICTION")
        # print(predictions)
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

    def get_data_generator_train_aug(self, type, batch_size=None, reduction=0.0):
        if not batch_size:
            batch_size = self.batch_size
        df = pandas.read_csv(os.path.join(self.path, type, "dataframe.csv"), dtype=str)
        return DataGenerator(df, 'image_path', 'label',
                             reduction=reduction,
                             image_size=int(self.w_conf["img_size"]),
                             aug_sequence=get_augmenting_sequence(self.w_conf) if self.w_conf[
                                                                                      "aug"] == 1 else get_no_augmenting_sequence(
                                 self.w_conf),
                             batch_size=batch_size)

    def get_data_generator(self, type, batch_size=None, split=0.0):
        if not batch_size:
            batch_size = self.batch_size
        return ImageDataGenerator(validation_split=split).flow_from_directory(
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
    ap.add_argument("--epochs", "-e", default=15, type=int)
    ap.add_argument("--tf_layers", "-t", default=150, type=int)
    ap.add_argument("--img_size", "-s", default=224, type=int)
    ap.add_argument("--steps_per_epoch", default=500, type=int)
    ap.add_argument("--aug", default=1, type=int)
    ap.add_argument("--grayscale", default=0, type=int)
    ap.add_argument("--model", "-m", default="InceptionResNetV2",
                    choices={"EfficientNetB0", "InceptionResNetV2", "InceptionV3", "ResNet50V2", "EfficientNetB5",
                             "EfficientNetB3",
                             "VGG16"})

    return ap.parse_args()


def main(args):
    global CLASSES
    CLASSES = os.listdir(os.path.join(args.dataset, "val"))
    print(len(CLASSES))
    CLASSES = [x for x in CLASSES if "." not in x]
    print(len(CLASSES))
    dataset_id = args.dataset.split("/")[-2],
    w_conf = {
        "architecture": args.model,
        "dataset_id": dataset_id[0],
        "dataset": args.dataset,
        "epochs": args.epochs,
        "tf_layers": args.tf_layers,
        "img_size": args.img_size,
        "model": args.model,
        "steps_per_epoch": args.steps_per_epoch,
        "aug": args.aug,
        "grayscale": args.grayscale
    }
    wandb.init(config=w_conf,
               name=F"ds: {dataset_id[0]} s: {args.img_size}px m: {args.model}, epok: {args.epochs}, w. TF: {args.tf_layers}",
               project="praca_magisterska")
    DATASET_PATH = os.path.abspath(os.path.join(args.dataset))

    dataSet = DataSet(DATASET_PATH, BATCH_SIZE, (args.img_size, args.img_size), w_conf)
    network = TFLegoClassifier(dataSet=dataSet)
    # network.prepare_model(Inception)
    modelCls = locate(F"lego_sorter_server.analysis.classification.models.models.{args.model}model")

    network.prepare_model(modelCls, args.img_size, tf_layers=args.tf_layers)
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
