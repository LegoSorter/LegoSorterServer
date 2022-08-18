import os
import threading
import time
from loguru import logger
import torch
import numpy
from pathlib import Path

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector



class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class YoloLegoDetectorOnnx(LegoDetector, metaclass=ThreadSafeSingleton):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "detection", "models", 
                                            "onnx",
                                            "last.onnx",
                                            # "yolo_model",
                                            #    "yolov5_medium_extended.pt"
                                               )):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()

    def __initialize__(self):
        if self.__initialized:
            raise Exception("YoloLegoDetectorOnnx already initialized")

        if not self.model_path.exists():
            logger.error(f"[YoloLegoDetectorOnnx] No model found in {str(self.model_path)}")
            raise RuntimeError(f"[YoloLegoDetectorOnnx] No model found in {str(self.model_path)}")

        start_time = time.time()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.model_path))

        if torch.cuda.is_available():
            self.model.cuda()
        elapsed_time = time.time() - start_time

        logger.info("[YoloLegoDetectorOnnx] Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True



    @staticmethod
    def xyxy2yxyx_scaled(xyxy):
        """
        returns (ymin, xmin, ymax, xmax)
        """
        return numpy.array([[coord[1], coord[0], coord[3], coord[2]] for coord in xyxy])

    @staticmethod
    def convert_results_to_common_format(results) -> DetectionResults:
        image_predictions = results.xyxyn[0].cpu().numpy()
        scores = image_predictions[:, 4]
        classes = image_predictions[:, 5].astype(numpy.int64) + 1
        boxes = YoloLegoDetectorOnnx.xyxy2yxyx_scaled(image_predictions[:, :4])

        return DetectionResults(detection_scores=scores, detection_classes=classes, detection_boxes=boxes)

    def detect_lego(self, image: numpy.ndarray) -> DetectionResults:
        if not self.__initialized:
            logger.info("YoloLegoDetectorOnnx is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        logger.info("[YoloLegoDetectorOnnx][detect_lego] Detecting bricks...")
        start_time = time.time()
        results = self.model([image.astype], size=image.shape[0])
        elapsed_time = 1000 * (time.time() - start_time)
        logger.info(f"[YoloLegoDetectorOnnx][detect_lego] Detecting bricks took {elapsed_time} milliseconds")

        return self.convert_results_to_common_format(results)
