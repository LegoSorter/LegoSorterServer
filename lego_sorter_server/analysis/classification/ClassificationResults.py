class ClassificationResults:
    def __init__(self, classification_classes, classification_scores):
        self.classification_classes = classification_classes
        self.classification_scores = classification_scores

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
