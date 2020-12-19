from collections import deque
from PIL.Image import Image


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageProcessingQueue(metaclass=Singleton):
    """Stores lego images for processing. Format of returned objects is a tuple (image, lego_class)"""
    def __init__(self, limit=1000):
        self.limit = limit
        self.in_memory_store = deque([])

    def next(self):
        return self.in_memory_store.popleft()

    def add(self, image: Image, lego_class='unknown'):
        self._check_limit()
        self.in_memory_store.append((image, lego_class))

    def len(self):
        return len(self.in_memory_store)

    def _check_limit(self):
        if len(self.in_memory_store) >= self.limit:
            raise Exception("Queue out of bound!")
