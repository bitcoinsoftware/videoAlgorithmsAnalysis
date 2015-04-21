import cv2
from AbstractClassifier import *

class SVM(AbstractClassifier):
    def __init__(self):
        self.classifier =cv2.SVM()
        self.params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )

    def train(self, samples, responses):
        self.classifier.train(np.array(samples, dtype=np.float32),np.array(responses, dtype=np.float32), params=self.params)