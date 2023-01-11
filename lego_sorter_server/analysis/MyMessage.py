class MyMessage:
    def __init__(self, ymin = None, xmin = None, ymax = None, xmax = None, label = None, score = None,label_top5 = None, score_top5 = None, id = None, session = None):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax
        self.label = label
        self.score = score
        self.label_top5 = label_top5
        self.score_top5 = score_top5
        self.id = id
        self.session = session
