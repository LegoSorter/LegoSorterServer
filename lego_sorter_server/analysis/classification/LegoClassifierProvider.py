from lego_sorter_server.analysis.classification.classifiers.TFLegoClassifier import TFLegoClassifier


class LegoClassifierProvider:
    classifier = TFLegoClassifier()

    @staticmethod
    def get_default_classifier():
        return LegoClassifierProvider.classifier
