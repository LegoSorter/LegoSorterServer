import os
import threading
import time
import numpy as np
import tensorflow as tf
from pathlib import Path


class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LegoDetector(metaclass=ThreadSafeSingleton):

    def __init__(self, model_path=os.path.join("lego_sorter_server", "detection", "models", "lego_detection_model", "saved_model")):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()

    @staticmethod
    def prepare_input_tensor(image):
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor

    def __initialize__(self):
        if self.__initialized:
            raise Exception("LegoDetector already initialized")

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        start_time = time.time()
        self.model = tf.saved_model.load(str(self.model_path))
        elapsed_time = time.time() - start_time

        print("Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    def detect_lego(self, image):
        if not self.__initialized:
            print("LegoDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        input_tensor = self.prepare_input_tensor(image)
        detections = self.model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections
