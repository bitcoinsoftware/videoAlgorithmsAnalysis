import cv2
from AbstractClassifier import *

class RTree(AbstractClassifier):
    def __init__(self):
        self.classifier =cv2.RTrees()

    def train(self, samples, responses):
        tflag=cv2.CV_ROW_SAMPLE
        self.classifier.train(np.array(samples, dtype=np.float32),tflag, np.array(responses, dtype=np.float32))