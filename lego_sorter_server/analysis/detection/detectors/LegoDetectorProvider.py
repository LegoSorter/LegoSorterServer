from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.analysis.detection.detectors.TFLegoDetector import TFLegoDetector
from lego_sorter_server.analysis.detection.detectors.YoloLegoDetector import YoloLegoDetector


class LegoDetectorProvider:

    @staticmethod
    def get_default_detector() -> LegoDetector:
        return TFLegoDetector()
