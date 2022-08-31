import threading

from lego_sorter_server.analysis.classification.classifiers.KerasClassifier import KerasClassifier
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.classifiers.TFLegoClassifier import TFLegoClassifier


class LegoClassifierProvider:
    __lock = threading.Lock()

    classifier: LegoClassifier = None

    @staticmethod
    def get_default_classifier() -> LegoClassifier:
        if not LegoClassifierProvider.classifier:
            LegoClassifierProvider.__lock.acquire()
            if not LegoClassifierProvider.classifier:
                LegoClassifierProvider.classifier = KerasClassifier()
                LegoClassifierProvider.classifier.load_model()
            LegoClassifierProvider.__lock.release()

        return LegoClassifierProvider.classifier
