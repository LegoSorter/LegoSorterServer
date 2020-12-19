from PIL import Image
from collections import deque


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ImageProcessingQueue(metaclass=Singleton):
    """Stores lego images for processing. Format of returned objects is {image, lego_class}"""
    def __init__(self, limit=1000):
        self.limit = limit
        self.in_memory_store = deque([])

    @staticmethod
    def rotate_image(image, rotation):
        if rotation == 90:
            image = image.transpose(Image.ROTATE_270)
        if rotation == 180:
            image = image.rotate(180)
        if rotation == 270:
            image = image.transpose(Image.ROTATE_90)

        return image

    def next(self):
        return self.in_memory_store.popleft()

    def add(self, image, lego_class='unknown', rotation=0):
        self._check_limit()
        image = self.rotate_image(image, rotation)
        self.in_memory_store.append({image, lego_class})

    def _check_limit(self):
        if len(self.in_memory_store) >= self.limit:
            raise Exception("Queue out of bound!")
