import logging
from typing import List, Tuple

from PIL.Image import Image

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.sorter.LegoSorterController import LegoSorterController
from lego_sorter_server.sorter.ordering.SimpleOrdering import SimpleOrdering


class SortingProcessor:
    def __init__(self):
        self.analysis_service: AnalysisService = AnalysisService()
        self.sorter_controller: LegoSorterController = LegoSorterController()
        self.ordering: SimpleOrdering = SimpleOrdering()

    @staticmethod
    def get_best_result(results):
        # TODO - max score, average score, max count?
        return results[0]

    def process_next_image(self, image: Image):
        current_results = self._process(image)

        self.ordering.process_current_results(current_results)

        while self.ordering.get_count_of_results_to_send() > 0:
            # Clear out the queue of processed bricks
            self._send_results_to_controller()

        return self.ordering.get_current_state()

    def _send_results_to_controller(self):
        processed_brick = self.ordering.pop_first_processed_brick()

        if len(processed_brick) == 0:
            return False

        best_result = self.get_best_result(processed_brick)
        logging.info(f"[SortingProcessor] Got the best result {best_result}. Returning the results...")
        self.sorter_controller.on_brick_recognized(best_result)

    def _process(self, image: Image) -> List[Tuple]:
        """
        Returns a list of recognized bricks ordered by the position on the belt - ymin desc
        """
        results = self.analysis_service.detect_and_classify(image)

        detected_count = len(results[0].detection_classes)
        if detected_count == 0:
            return [()]

        logging.info(f"[SortingProcessor] Detected a lego brick, processing...")

        if detected_count > 1:
            logging.warning(f"[SortingProcessor] More than one brick detected '(detected_count = {detected_count}), "
                            f"there should be only one brick on the tape at the same time.")

        zipped_results = list(zip(results[0].detection_boxes,
                                  results[1].classification_classes,
                                  results[1].classification_scores))

        return self.order_by_bounding_box_position(zipped_results)

    @staticmethod
    def order_by_bounding_box_position(zipped_results: List[Tuple[Tuple, str, float]]) -> List[Tuple]:
        # sort by ymin
        return sorted(zipped_results, key=lambda res: res[0][0], reverse=True)
