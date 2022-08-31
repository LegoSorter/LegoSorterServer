import threading

from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
# from lego_sorter_server.analysis.detection.detectors.TFLegoDetector import TFLegoDetector
# from lego_sorter_server.analysis.detection.detectors.DeepSparseDetector import DeepSparseDetector
# from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
# from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorEdgeTpu import YoloLegoDetectorEdgeTpu
# from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorOnnx import YoloLegoDetectorOnnx

# 0 - YOLOv5
# 1 - YOLOv5 run in DeepSparse
# 2 - YOLOv5 Google Coral Edge TPU classify 3 parts of scaled image
# 3 - YOLOv5 run in Onnx
# 4 - YOLOv5 Google Coral Edge TPU classify 3 parts of scaled and cropped image
import os
from sys import platform


if os.getenv("LEGO_SORTER_DETECTOR") == "0":
    from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
elif os.getenv("LEGO_SORTER_DETECTOR") == "1" and platform == "linux":
    try:
        from lego_sorter_server.analysis.detection.detectors.DeepSparseDetector import DeepSparseDetector
    except ImportError:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
        os.environ["LEGO_SORTER_DETECTOR"] = "0"
elif os.getenv("LEGO_SORTER_DETECTOR") == "2" and platform == "linux":
    # Windows doesn't work with newest version https://github.com/google-coral/edgetpu/issues/468
    try:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorEdgeTpu import YoloLegoDetectorEdgeTpu
    except ImportError:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
        os.environ["LEGO_SORTER_DETECTOR"] = "0"
elif os.getenv("LEGO_SORTER_DETECTOR") == "3":
    try:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorOnnx import YoloLegoDetectorOnnx
    except ImportError:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
        os.environ["LEGO_SORTER_DETECTOR"] = "0"
elif os.getenv("LEGO_SORTER_DETECTOR") == "4" and platform == "linux":
    # Windows doesn't work with newest version https://github.com/google-coral/edgetpu/issues/468
    try:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorEdgeTpuSimple import YoloLegoDetectorEdgeTpuSimple
    except ImportError:
        from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
        os.environ["LEGO_SORTER_DETECTOR"] = "0"
else:
    from lego_sorter_server.analysis.detection.detectors.YoloLegoDetectorFast import YoloLegoDetectorFast
    os.environ["LEGO_SORTER_DETECTOR"] = "0"


class LegoDetectorProviderFast:
    __lock = threading.Lock()
    __detector: LegoDetector = None

    @staticmethod
    def get_default_detector() -> LegoDetector:
        if not LegoDetectorProviderFast.__detector:
            LegoDetectorProviderFast.__lock.acquire()
            if not LegoDetectorProviderFast.__detector:
                if os.getenv("LEGO_SORTER_DETECTOR") == "0":
                    LegoDetectorProviderFast.__detector = YoloLegoDetectorFast()
                elif os.getenv("LEGO_SORTER_DETECTOR") == "1":
                    LegoDetectorProviderFast.__detector = DeepSparseDetector()
                elif os.getenv("LEGO_SORTER_DETECTOR") == "2":
                    LegoDetectorProviderFast.__detector = YoloLegoDetectorEdgeTpu()
                elif os.getenv("LEGO_SORTER_DETECTOR") == "3":
                    LegoDetectorProviderFast.__detector = YoloLegoDetectorOnnx()
                elif os.getenv("LEGO_SORTER_DETECTOR") == "4":
                    LegoDetectorProviderFast.__detector = YoloLegoDetectorEdgeTpuSimple()
                else:
                    LegoDetectorProviderFast.__detector = YoloLegoDetectorFast()
                LegoDetectorProviderFast.__detector.__initialize__()
            LegoDetectorProviderFast.__lock.release()
        return LegoDetectorProviderFast.__detector
