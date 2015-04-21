import cv2
from AbstractClassifier import *

class KNearest(AbstractClassifier):
    classifier =cv2.KNearest()

    def __init__(self, param=1):
        self.kNumber = param

    def predict(self, sample):
        return self.classifier.find_nearest(np.matrix(sample),int(self.kNumber))[0]