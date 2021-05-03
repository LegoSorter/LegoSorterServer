from PIL import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults


class LegoClassifier:
    def predict(self, images: [Image.Image]) -> ClassificationResults:
        pass
