import threading
import os

from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
# from lego_sorter_server.analysis.classification.classifiers.TFLegoClassifier import TFLegoClassifier

# 0 - Keras
# 1 - TinyVit
if os.getenv("LEGO_SORTER_CLASSIFIER") == "0":
    try:
        from lego_sorter_server.analysis.classification.classifiers.KerasClassifierFast import KerasClassifierFast
    except ImportError:
        from lego_sorter_server.analysis.classification.classifiers.TinyViTClassifierFast import TinyViTClassifier
        os.environ["LEGO_SORTER_CLASSIFIER"] = "1"
elif os.getenv("LEGO_SORTER_CLASSIFIER") == "1":
    try:
        from lego_sorter_server.analysis.classification.classifiers.TinyViTClassifierFast import TinyViTClassifier
    except ImportError:
        from lego_sorter_server.analysis.classification.classifiers.KerasClassifierFast import KerasClassifierFast
        os.environ["LEGO_SORTER_CLASSIFIER"] = "0"
else:
    from lego_sorter_server.analysis.classification.classifiers.KerasClassifierFast import KerasClassifierFast
    os.environ["LEGO_SORTER_CLASSIFIER"] = "0"


class LegoClassifierProviderFast:
    __lock = threading.Lock()

    classifier: LegoClassifier = None

    @staticmethod
    def get_default_classifier() -> LegoClassifier:
        if not LegoClassifierProviderFast.classifier:
            LegoClassifierProviderFast.__lock.acquire()
            if not LegoClassifierProviderFast.classifier:
                if os.getenv("LEGO_SORTER_CLASSIFIER") == "0":
                    LegoClassifierProviderFast.classifier = KerasClassifierFast()
                elif os.getenv("LEGO_SORTER_CLASSIFIER") == "1":
                    LegoClassifierProviderFast.classifier = TinyViTClassifier()
                else:
                    LegoClassifierProviderFast.classifier = TinyViTClassifier()
                LegoClassifierProviderFast.classifier.load_model()
            LegoClassifierProviderFast.__lock.release()

        return LegoClassifierProviderFast.classifier
