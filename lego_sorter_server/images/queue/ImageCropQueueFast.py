import os
from collections import deque
from typing import Tuple, List, Optional

import numpy

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
# from PIL.Image import Image

from lego_sorter_server.generated.LegoAnalysisFast_pb2 import FastImageRequest


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


SORTER_TAG = "sorter"
CAPTURE_TAG = "capture"


class ImageCropQueueFast(metaclass=Singleton):
    """Stores lego images for processing. Format of returned objects is a tuple (image, lego_class)"""

    def __init__(self, limit=None):
        self.limit = limit
        if self.limit is None:
            self.limit = int(os.getenv("LEGO_SORTER_CROPS_QUEUE_LIMIT"))
        self.in_memory_stores = {SORTER_TAG: deque([], maxlen=self.limit), CAPTURE_TAG: deque([], maxlen=self.limit)}

    def next(self, tag: str) -> Tuple[List[numpy.ndarray], DetectionResults, ClassificationResults, Optional[int], str, str]:
        return self.in_memory_stores.get(tag).popleft()

    def add(self, tag: str, crops: [numpy.ndarray], detections: DetectionResults, classifications: ClassificationResults, imageid: Optional[int], id:str, lego_class='unknown') -> None:
        # self._check_limit(tag)
        self.in_memory_stores.get(tag).append((crops, detections, classifications, imageid, id, lego_class))

    def len(self, tag: str) -> int:
        return len(self.in_memory_stores.get(tag))

    def clear(self, tag: str) -> None:
        self.in_memory_stores.get(tag).clear()

    def is_full(self, tag: str):
        return len(self.in_memory_stores.get(tag)) >= self.limit

    def _check_limit(self, tag: str) -> None:
        if len(self.in_memory_stores.get(tag)) > self.limit:
            raise Exception("Queue out of bound!")
