from abc import ABC, abstractmethod


class Transformation(ABC):
    @staticmethod
    @abstractmethod
    def transform(img):
        ...
