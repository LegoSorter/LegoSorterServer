from typing import List


class ClassificationResults:
    def __init__(self, classification_classes: List[str], classification_scores: List[float], classification_classes_top5: List[List[str]] = None, classification_scores_top5: List[List[float]] = None):
        self.classification_classes = classification_classes
        self.classification_scores = classification_scores
        self.classification_classes_top5 = classification_classes_top5
        self.classification_scores_top5 = classification_scores_top5

    @staticmethod
    def from_dict(classification_results_dict):
        return ClassificationResults(classification_results_dict['classification_classes'],
                                     classification_results_dict['classification_scores'])

    @staticmethod
    def empty():
        return ClassificationResults([], [])

    def get_as_dict(self):
        return {'classification_classes', self.classification_classes,
                'classification_scores', self.classification_scores}

    def get_as_dict_top5(self):
        return {'classification_classes_top5', self.classification_classes_top5,
                'classification_classes_top5', self.classification_classes_top5}
