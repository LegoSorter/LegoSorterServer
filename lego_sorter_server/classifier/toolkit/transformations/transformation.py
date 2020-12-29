from abc import ABC, abstractmethod


class TransformationException(Exception):
    def __init__(self, *args, **kwargs):
        if "prefix" in kwargs:
            self.prefix = kwargs["prefix"]
        else:
            self.prefix = ""

class Transformation(ABC):
    @staticmethod
    @abstractmethod
    def transform(img):
        ...
