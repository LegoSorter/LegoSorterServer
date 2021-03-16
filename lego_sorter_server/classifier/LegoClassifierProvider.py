class Fake:
    def classify(self, anything):
        pass


def get_default_classifier():
    return Fake()
