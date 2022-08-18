import os
import threading
import time
from loguru import logger
import torch
import numpy
from pathlib import Path

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector


from inspect import getmembers, isfunction

from deepsparse import compile_model, cpu
from deepsparse.utils import verify_outputs
from sparsezoo.models import detection
from sparsezoo.objects import Model

from typing import Any, List, Union


from deepsparse import compile_model
from lego_sorter_server.analysis.detection.detectors.deepsparse_utils import (
    YoloPostprocessor,
    annotate_image,
    download_pytorch_model_if_stub,
    get_yolo_loader_and_saver,
    modify_yolo_onnx_input_shape,
    postprocess_nms,
    yolo_onnx_has_postprocessing,
)
from sparseml.onnx.utils import override_model_batch_size

class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DeepSparseDetector(LegoDetector, metaclass=ThreadSafeSingleton):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "detection", "models", 
                                                "onnx",
                                                "best.onnx",#nowy
                                                # "last.onnx",#pierwszy
                                                # "yolo_model",
                                                # "last.pt")):
                                                # "yolov5_medium_extended.pt"
                                                )):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()

        # CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

        # model_registry = dict(getmembers(detection, isfunction))

    def _load_model(self, model_path, image_shape, num_cores=0) -> Any:

        model_filepath, _ = modify_yolo_onnx_input_shape(model_path, image_shape)
        has_postprocessing = yolo_onnx_has_postprocessing(model_filepath)

        # load model

        logger.info(f"Compiling DeepSparse model for {model_filepath}")
        model = compile_model(model_filepath, 1, num_cores)

        return model, has_postprocessing

    def __initialize__(self):
        if self.__initialized:
            raise Exception("DeepSparseDetector already initialized")

        if not self.model_path.exists():
            logger.error(f"[DeepSparseDetector] No model found in {str(self.model_path)}")
            raise RuntimeError(f"[DeepSparseDetector] No model found in {str(self.model_path)}")

        start_time = time.time()
        self.model, has_postprocessing = DeepSparseDetector._load_model(self, model_path=str(self.model_path), image_shape=(640, 640))

        self.postprocessor = (
            YoloPostprocessor((640, 640))
            if not has_postprocessing
            else None
        )

        self.engine = compile_model(str(self.model_path), batch_size=1, num_cores=0)
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(self.model_path))
        # if torch.cuda.is_available():
        #     self.model.cuda()
        elapsed_time = time.time() - start_time

        logger.info("[DeepSparseDetector] Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True

    @staticmethod
    def xyxy2yxyx_scaled(xyxy):
        """
        returns (ymin, xmin, ymax, xmax)
        """
        return numpy.array([[coord[1]/640, coord[0]/640, coord[3]/640, coord[2]/640] for coord in xyxy])

    @staticmethod
    def convert_results_to_common_format(results) -> DetectionResults:
        image_predictions = results
        scores = image_predictions[:, 4]
        classes = image_predictions[:, 5].astype(numpy.int64) + 1
        boxes = DeepSparseDetector.xyxy2yxyx_scaled(image_predictions[:, :4])

        return DetectionResults(detection_scores=scores, detection_classes=classes, detection_boxes=boxes)

    def _preprocess_batch(self, batch: numpy.ndarray) -> Union[numpy.ndarray, torch.Tensor]:
        if len(batch.shape) == 3:
            batch = numpy.moveaxis(batch, -1, 0)
            batch = batch.reshape(1, *batch.shape)
            batch = numpy.ascontiguousarray(batch)
        else:
            batch = numpy.ascontiguousarray(batch)
        return batch

    def _run_model(
            self, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
    ) -> List[Union[numpy.ndarray, torch.Tensor]]:
        outputs = None
        # deepsparse
        outputs = model.run([batch])

        return outputs

    def detect_lego(self, image: numpy.ndarray) -> DetectionResults:
        if not self.__initialized:
            logger.info("DeepSparseDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        logger.info("[DeepSparseDetector][detect_lego] Detecting bricks...")
        start_time = time.time()
        # results = self.model([image], size=image.shape[0])
        # inp = [numpy.random.rand(1, 3, 640, 640).astype(numpy.uint8)]
        # image = numpy.moveaxis(image, -1, 0)
        # image = numpy.expand_dims(image, axis=0)
        # image = [image.astype(numpy.uint8)]
        # results, elapsed_time = self.engine.timed_run(image)

        # pre-processing
        batch = self._preprocess_batch(image)

        # inference
        outputs = self._run_model(self.model, batch)

        # post-processing
        if self.postprocessor:
            outputs = self.postprocessor.pre_nms_postprocess(outputs)
        else:
            outputs = outputs[0]  # post-processed values stored in first output

        # NMS
        outputs = postprocess_nms(outputs)[0]

        elapsed_time = 1000 * (time.time() - start_time)
        logger.info(f"[DeepSparseDetector][detect_lego] Detecting bricks took {elapsed_time} milliseconds")

        return self.convert_results_to_common_format(outputs)
