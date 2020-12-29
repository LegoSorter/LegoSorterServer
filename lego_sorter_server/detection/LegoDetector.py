import os
import threading
import time
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path

from lego_sorter_server.detection import DetectionUtils
from lego_sorter_server.detection.DetectionUtils import crop_with_margin


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

        start_time = time.time()
        self.model = tf.saved_model.load(str(self.model_path))
        elapsed_time = time.time() - start_time

        logging.info("Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    def detect_lego(self, image):
        if not self.__initialized:
            logging.info("LegoDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        input_tensor = self.prepare_input_tensor(image)
        detections = self.model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections

    def detect_and_crop(self, image):
        width, height = image.size
        image_resized, scale = DetectionUtils.resize(image, 640)
        detections = self.detect_lego(np.array(image_resized))
        detected_counter = 0
        new_images = []
        for i in range(100):
            if detections['detection_scores'][i] < 0.5:
                break  # IF SORTED

            detected_counter += 1
            ymin, xmin, ymax, xmax = [int(i * 640 * 1 / scale) for i in detections['detection_boxes'][i]]

            # if bb is out of bounds
            if ymax >= height or xmax >= width:
                continue

            new_images += [crop_with_margin(image, ymin, xmin, ymax, xmax)]

        return new_images
