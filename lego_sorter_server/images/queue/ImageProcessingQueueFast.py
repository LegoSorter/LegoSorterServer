from typing import Tuple, Optional
from collections import deque

import numpy
from PIL.Image import Image

from lego_sorter_server.database.Models import DBImage


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


SORTER_TAG = "sorter"
CAPTURE_TAG = "capture"


class ImageProcessingQueueFast(metaclass=Singleton):
    """Stores lego images for processing. Format of returned objects is a tuple (image, lego_class)"""

    def __init__(self, limit=1000):
        self.limit = limit
        self.in_memory_stores = {SORTER_TAG: deque([], maxlen=limit), CAPTURE_TAG: deque([], maxlen=limit)}

    # def next(self, tag: str) -> Tuple[Image, Optional[int]]:
    def next(self, tag: str) -> Tuple[numpy.ndarray, Optional[int]]:
        return self.in_memory_stores.get(tag).pop()

    # def add(self, tag: str, image: Image, imageid: Optional[int]) -> None:
    def add(self, tag: str, image: numpy.ndarray, imageid: Optional[int]) -> None:
        # self._check_limit(tag)
        self.in_memory_stores.get(tag).append((image, imageid))

    def len(self, tag: str) -> int:
        return len(self.in_memory_stores.get(tag))

    def clear(self, tag: str) -> None:
        self.in_memory_stores.get(tag).clear()

    def is_full(self, tag: str):
        return len(self.in_memory_stores.get(tag)) >= self.limit

    def _check_limit(self, tag: str) -> None:
        if len(self.in_memory_stores.get(tag)) > self.limit:
            raise Exception("Queue out of bound!")
