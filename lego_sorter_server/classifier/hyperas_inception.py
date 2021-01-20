import os
import argparse
from datetime import datetime
from pydoc import locate

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# inceptionv3 imports
from PIL import Image
# callback imports
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

#from models.models import *
#from lego_sorter_server.classifier.models.inceptionClear import InceptionClear
from lego_sorter_server.classifier.toolkit.transformations.simple import Simple
from lego_sorter_server.connection.KaskServerConnector import KaskServerConnector

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 32
IMG_SIZE = (299, 299)
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join("lego_sorter_server", "classifier", "models", "saved"))
DATASET_PATH = os.path.abspath(os.path.join("images/dataset"))

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


def model(dataSet):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense({{choice([128, 256, 512, 1024, 2048])}}, activation='relu')(x)

    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers[:-2]:
        layer.trainable = False
    opt = Adam(lr={{uniform(0,1)}})
    model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=[tf.keras.metrics.CategoricalAccuracy()])
    train_generator = dataSet.get_data_generator("train")
    validation_generator = dataSet.get_data_generator("val")
    test_generator = dataSet.get_data_generator("test",1)
    model.fit(train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator))
    acc = model.evaluate(test_generator, steps=20)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



#d

def DataSet_func():
    BATCH_SIZE = 32
    IMG_SIZE = (299, 299)
    DATASET_PATH = os.path.abspath(os.path.join("images/dataset"))
    class DataSet:
        def __init__(self, path, batch_size, img_size):
            self.path = path
            self.batch_size = batch_size
            self.img_size = img_size

        def get_data_generator(self, type, batch_size=None):
            if not batch_size:
                batch_size = self.batch_size
            generator =  ImageDataGenerator(rescale=1. / 255, rotation_range=15, horizontal_flip=True,vertical_flip=True).flow_from_directory(os.path.join(self.path, type), target_size=self.img_size, shuffle=True, batch_size=batch_size, class_mode='categorical')
            return generator

    dataSet = DataSet(DATASET_PATH, BATCH_SIZE, IMG_SIZE)
    return dataSet

def main():

    #dataSet = DataSet(DATASET_PATH, BATCH_SIZE, IMG_SIZE)
    best_run, best_model = optim.minimize(model=model,
                                      data=DataSet_func,
                                      algo=tpe.suggest,
                                      max_evals=30,
                                      trials=Trials())

if __name__ == '__main__':
    main()
