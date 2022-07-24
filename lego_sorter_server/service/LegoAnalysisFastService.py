from lego_sorter_server.analysis.AnalysisFastService import AnalysisFastService
import logging
import time

from lego_sorter_server.generated import LegoAnalysisFast_pb2_grpc
from lego_sorter_server.generated.Messages_pb2 import ImageRequest, ListOfBoundingBoxes
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils


class LegoAnalysisFastService(LegoAnalysisFast_pb2_grpc.LegoAnalysisFastServicer):
    def __init__(self):
        self.analysis_service = AnalysisFastService()

    def DetectBricks(self, request: ImageRequest, context):
        logging.info("[LegoAnalysisService] Request received, processing...")
        start_time = time.time()

        detection_results = self._detect_bricks(request)
        bbs_list = ImageProtoUtils.prepare_bbs_response_from_detection_results(detection_results)

        elapsed_millis = int((time.time() - start_time) * 1000)
        logging.info(f"[LegoAnalysisService] Detecting and preparing response took {elapsed_millis} milliseconds.")

        return bbs_list

    def DetectAndClassifyBricks(self, request: ImageRequest, context):
        logging.info("[LegoAnalysisService] Request received, processing...")
        start_time = time.time()

        image = ImageProtoUtils.prepare_image(request)
        detection_results, classification_results = self.analysis_service.detect_and_classify(image)
        bb_list = ImageProtoUtils.prepare_response_from_analysis_results(detection_results, classification_results)

        elapsed_millis = (time.time() - start_time) * 1000
        logging.info(f"[LegoAnalysisService] Detecting, classifying and preparing response took "
                     f"{elapsed_millis} milliseconds.")

        return bb_list

    def _detect_bricks(self, request: ImageRequest) -> ListOfBoundingBoxes:
        image = ImageProtoUtils.prepare_image(request)
        detection_results = self.analysis_service.detect(image)

        return detection_results
