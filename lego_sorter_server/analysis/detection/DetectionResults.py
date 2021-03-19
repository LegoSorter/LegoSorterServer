class DetectionResults:
    def __init__(self, detection_scores, detection_classes, detection_boxes):
        self.detection_scores = detection_scores
        self.detection_classes = detection_classes
        self.detection_boxes = detection_boxes  # (ymin, xmin, ymax, xmax)

    @classmethod
    def from_dict(cls, results_dict):
        return DetectionResults(results_dict['detection_scores'],
                                results_dict['detection_classes'],
                                results_dict['detection_boxes'])

    def get_as_dict(self):
        return {'detection_scores': self.detection_scores,
                'detection_classes': self.detection_classes,
                'detection_boxes': self.detection_boxes}
