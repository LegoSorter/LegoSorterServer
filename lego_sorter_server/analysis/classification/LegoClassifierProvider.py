import threading

from lego_sorter_server.analysis.classification.classifiers.KerasClassifier import KerasClassifier
from lego_sorter_server.analysis.classification.classifiers.KerasLabelClassifier import KerasLabelClassifier
from lego_sorter_server.analysis.classification.classifiers.KerasOnnxLabelClassifier import KerasOnnxLabelClassifier
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
                ## Base solution based on unmodified h5 model -> returns one class with the highiest probability 
                #LegoClassifierProvider.classifier = KerasClassifier()
                
                ## Base solution based on unmodified h5 model -> returns multiple classes with the highiest probabilities
                #LegoClassifierProvider.classifier = KerasLabelClassifier()

                ## Modified solution based on converted h5 model to ONNX format -> returns multiple classes with the highiest probabilities
                LegoClassifierProvider.classifier = KerasOnnxLabelClassifier()

                LegoClassifierProvider.classifier.load_model()
            LegoClassifierProvider.__lock.release()

        return LegoClassifierProvider.classifier