import cv2
from AbstractClassifier import *

class GBTree(AbstractClassifier):
    def __init__(self, param = None):
        self.classifier =cv2.GBTrees()

    def train(self, samples, responses):
        tflag=cv2.CV_ROW_SAMPLE
        self.classifier.train(np.array(samples, dtype=np.float32), tflag, np.array(responses, dtype=np.float32))

    def predict(self, sample):
        return round(self.classifier.predict(np.matrix(sample)))