import os
import threading
import time
import logging
import torch
import numpy
from pathlib import Path

from lego_sorter_server.connection.KaskServerConnector import KaskServerConnector


class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class YoloLegoDetector(metaclass=ThreadSafeSingleton):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "detection", "models", "yolo_model", "yolov5_small.pt")):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()

    def __initialize__(self):
        if self.__initialized:
            raise Exception("YoloLegoDetector already initialized")

        if not self.model_path.exists():
            KaskServerConnector().download_models()

        start_time = time.time()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=str(self.model_path))
        elapsed_time = time.time() - start_time

        logging.info("Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    @staticmethod
    def xyxy2yxyx_scaled(xyxy, size):
        return numpy.array([numpy.array([coord[1], coord[0], coord[3], coord[2]]) / size for coord in xyxy])

    @staticmethod
    def convert_results_to_common_format(results, size):
        image_predictions = results.xyxy[0].numpy()
        detection_scores = image_predictions[:, 4]
        detection_classes = image_predictions[:, 5].astype(numpy.int64) + 1
        detection_boxes = YoloLegoDetector.xyxy2yxyx_scaled(image_predictions[:, :4], size)

        return {'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                'detection_boxes': detection_boxes}

    def detect_lego(self, image):
        if not self.__initialized:
            logging.info("YoloLegoDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        results = self.model([image], size=image.shape[0])
        return self.convert_results_to_common_format(results, image.shape[0])
