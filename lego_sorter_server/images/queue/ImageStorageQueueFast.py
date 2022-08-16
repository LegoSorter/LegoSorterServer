from collections import deque
from typing import Tuple
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


class ImageStorageQueueFast(metaclass=Singleton):
    """Stores lego images for processing. Format of returned objects is a tuple (image, lego_class)"""

    def __init__(self, limit=1000):
        self.limit = limit
        self.in_memory_stores = {SORTER_TAG: deque([], maxlen=limit), CAPTURE_TAG: deque([], maxlen=limit)}

    def next(self, tag: str) -> Tuple[FastImageRequest, str]:
        return self.in_memory_stores.get(tag).pop()

    def add(self, tag: str, image: FastImageRequest, lego_class='unknown') -> None:
        # self._check_limit(tag)
        self.in_memory_stores.get(tag).append((image, lego_class))

    def len(self, tag: str) -> int:
        return len(self.in_memory_stores.get(tag))

    def clear(self, tag: str) -> None:
        self.in_memory_stores.get(tag).clear()

    def is_full(self, tag: str):
        return len(self.in_memory_stores.get(tag)) >= self.limit

    def _check_limit(self, tag: str) -> None:
        if len(self.in_memory_stores.get(tag)) > self.limit:
            raise Exception("Queue out of bound!")
