from abc import ABC, abstractmethod


class Processor(ABC):
    @staticmethod
    @abstractmethod
    def precalc_sizes(src, classes, types, div_unit):
        ...

    @staticmethod
    @abstractmethod
    def calc_probs(candidates):
        ...

