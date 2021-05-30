from pathlib import Path
from typing import List

from PIL import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults


class LegoClassifier:
    def __init__(self):
        self.class_names: List[str] = self.read_classes_from_file()

    def predict(self, images: [Image.Image]) -> ClassificationResults:
        pass

    def read_classes_from_file(self, classes_file="./lego_sorter_server/analysis/classification/models/classes.txt") -> List[str]:
        with open(Path(classes_file)) as file:
            return [class_str.strip() for class_str in file]
