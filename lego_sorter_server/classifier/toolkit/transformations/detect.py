from lego_sorter_server.classifier.toolkit.transformations.transformation import Transformation, TransformationException
from lego_sorter_server.detection.detectors.TFLegoDetector import TFLegoDetector


class Detect(Transformation):
    @staticmethod
    def transform(img):
        legoDetector = TFLegoDetector()
        detected = legoDetector.detect_and_crop(img)
        if len(detected) != 1:
            raise TransformationException(F"Detected objects: {len(detected)} should be: 1",
                                          prefix=str(len(detected)))
        else:
            return detected[0]
