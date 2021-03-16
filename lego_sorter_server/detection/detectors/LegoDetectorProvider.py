from lego_sorter_server.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.detection.detectors.YoloLegoDetector import YoloLegoDetector


def get_default_detector() -> LegoDetector:
    return YoloLegoDetector()
