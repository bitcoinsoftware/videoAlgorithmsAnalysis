import cv2
from AbstractClassifier import *

class EM(AbstractClassifier):
    classifier =cv2.EM()

    def predict(self, sample):
        return self.classifier.predict(np.matrix(sample))[0][1]