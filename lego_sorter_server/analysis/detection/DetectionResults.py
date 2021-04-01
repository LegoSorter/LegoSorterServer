from typing import Tuple, List

DETECTION_SCORES_NAME = 'detection_scores'
DETECTION_CLASSES_NAME = 'detection_classes'
DETECTION_BOXES_NAME = 'detection_boxes'


class DetectionResults:

    def __init__(self, detection_scores, detection_classes, detection_boxes):
        assert len(detection_scores) == len(detection_classes) == len(detection_boxes)

        self.detection_scores: List[float] = detection_scores
        self.detection_classes: List[str] = detection_classes
        self.detection_boxes: List[Tuple[int, int, int, int]] = detection_boxes  # (ymin, xmin, ymax, xmax)

    @classmethod
    def from_dict(cls, results_dict):
        assert len(results_dict[DETECTION_SCORES_NAME]) \
               == len(results_dict[DETECTION_CLASSES_NAME]) \
               == len(results_dict[DETECTION_BOXES_NAME])

        return DetectionResults(results_dict[DETECTION_SCORES_NAME],
                                results_dict[DETECTION_CLASSES_NAME],
                                results_dict[DETECTION_BOXES_NAME])

    def get_as_dict(self):
        return {DETECTION_SCORES_NAME: self.detection_scores,
                DETECTION_CLASSES_NAME: self.detection_classes,
                DETECTION_BOXES_NAME: self.detection_boxes}
