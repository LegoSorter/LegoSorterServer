import logging
import os
import time

import tensorflow as tf
import numpy as np

from typing import List

from tensorflow import keras
from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class_names = ['11212',
               '14918',
               '15207',
               '15254',
               '15332',
               '15397',
               '20896',
               '2431',
               '2458',
               '28573',
               '2877',
               '28961',
               '29170',
               '2926',
               '3001',
               '3002',
               '3003',
               '3004',
               '3004 lub 35743 (transparent)',
               '30044',
               '3005',
               '3009',
               '3010',
               '30136',
               '30165',
               '3020',
               '3021',
               '3022',
               '3023',
               '3028',
               '3031',
               '30367',
               '3037',
               '30387',
               '3039',
               '3039 lub 35277 (transparent)',
               '3040',
               '30414',
               '3045',
               '3062',
               '3068',
               '3069',
               '3245',
               '3460',
               '35340',
               '3622',
               '3659',
               '3660',
               '3665',
               '3666',
               '3676',
               '3679',
               '3680',
               '3710',
               '3795',
               '3957',
               '4070',
               '4081',
               '4150',
               '44237',
               '4727',
               '47905',
               '4827',
               '50950',
               '54200',
               '58090',
               '6014',
               '60471',
               '60479',
               '60481',
               '60592',
               '60593',
               '60596',
               '60598',
               '60599',
               '60607',
               '60621',
               '6081',
               '6091',
               '6091 lub 32807',
               '61252',
               '6141',
               '6143 lub 39223 lub 6116 (transparent)',
               '6215',
               '6232',
               '64288',
               '85080',
               '85984',
               '87079',
               '87697',
               '90195',
               '92411',
               '92582',
               '93273',
               '98262',
               '99206']


class KerasClassifier(LegoClassifier):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "classification", "models",
                                               "keras_model", "residual_small.h5")):
        self.model_path = model_path
        self.initialized = False
        self.size = (180, 180)

    def __load_model(self):
        self.model = keras.models.load_model(self.model_path)
        self.initialized = True

    def predict(self, images: List[Image]) -> ClassificationResults:
        if not self.initialized:
            self.__load_model()

        if len(images) == 0:
            return ClassificationResults.empty()

        images_array = []
        start_time = time.time()
        for img in images:
            transformed = Simple.transform(img, self.size[0])
            img_array = np.array(transformed)
            img_array = np.expand_dims(img_array, axis=0)
            images_array.append(img_array)
        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        predictions = self.model(np.vstack(images_array))

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[KerasClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        indices = [int(np.argmax(values)) for values in predictions]
        classes = [class_names[index] for index in indices]
        scores = [prediction[index] for index, prediction in zip(indices, predictions)]

        return ClassificationResults(classes, scores)
