import cv2
from AbstractClassifier import *

class NormalBayesClassifier(AbstractClassifier):
    classifier =cv2.NormalBayesClassifier()

    def predict(self, sample):
        return self.classifier.predict(np.matrix(sample))[0]