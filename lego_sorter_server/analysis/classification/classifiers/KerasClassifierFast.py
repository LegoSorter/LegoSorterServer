import numpy
from loguru import logger
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


class KerasClassifierFast(LegoClassifier):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "classification", "models",
                                               "keras_model", "447_classes.h5")):
        super().__init__()
        env_model_path = os.getenv("LEGO_SORTER_KERAS_MODEL_PATH")
        if env_model_path is None or env_model_path == "":
            self.model_path = model_path
        else:
            self.model_path = env_model_path
        self.model = None
        self.initialized = False
        self.size = (224, 224)

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)
        self.initialized = True

    def predict(self, images: List[numpy.ndarray]) -> ClassificationResults:
        if not self.initialized:
            self.load_model()

        if len(images) == 0:
            return ClassificationResults.empty()

        images_array = []
        start_time = time.time()
        for img in images:
            transformed = Simple.transform_cv2(img, self.size[0])
            img_array = np.expand_dims(transformed, axis=0)
            images_array.append(img_array)
        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        predictions = self.model(np.vstack(images_array))

        predicting_elapsed_time_ms = 1000 * (time.time() - start_time) - processing_elapsed_time_ms

        logger.info(f"[KerasClassifierFast] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        indices = [int(np.argmax(values)) for values in predictions]
        predictions_np_array = np.array(predictions)
        indices_top5 = np.array([np.argpartition(values, -5)[-5:].tolist() for values in predictions])
        indices_top5_sorted = [index[np.argsort(predictions_np_array[i][index])][::-1] for i, index in enumerate(indices_top5)]
        classes = [self.class_names[index] for index in indices]
        classes_top5 = [[self.class_names[ind] for ind in index] for index in indices_top5_sorted]
        scores = [float(prediction[index]) for index, prediction in zip(indices, predictions)]
        scores_top5 = [[float(prediction[ind]) for ind in index] for index, prediction in zip(indices_top5_sorted, predictions)]

        return ClassificationResults(classes, scores, classes_top5, scores_top5)
