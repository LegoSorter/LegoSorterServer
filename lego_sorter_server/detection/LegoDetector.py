import time
import numpy as np

import tensorflow as tf


def prepare_input_tensor(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor


class LegoDetector:
    __initialized = False

    def __init__(self, model_path="./models/lego_detection_model/saved_model"):
        self.model_path = model_path

    def __initialize__(self):
        if self.__initialized:
            raise Exception("LegoDetector already initialized")

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        start_time = time.time()
        self.model = tf.saved_model.load(self.model_path)
        elapsed_time = time.time() - start_time

        print("Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    def detect_lego(self, image):
        if not self.__initialized:
            print("LegoDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        input_tensor = prepare_input_tensor(image)

        detections = self.model(input_tensor)
        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections
