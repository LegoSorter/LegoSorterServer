import threading

from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.analysis.detection.detectors.TFLegoDetector import TFLegoDetector
from lego_sorter_server.analysis.detection.detectors.YoloLegoDetector import YoloLegoDetector


class LegoDetectorProvider:
    __lock = threading.Lock()
    __detector: LegoDetector = None

    @staticmethod
    def get_default_detector() -> LegoDetector:
        if not LegoDetectorProvider.__detector:
            LegoDetectorProvider.__lock.acquire()
            if not LegoDetectorProvider.__detector:
                LegoDetectorProvider.__detector = YoloLegoDetector()
                LegoDetectorProvider.__detector.__initialize__()
            LegoDetectorProvider.__lock.release()
        return LegoDetectorProvider.__detector
