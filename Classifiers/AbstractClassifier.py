import numpy as np

class AbstractClassifier:
    #def __init__(self, param = None):

        #self.classifier = None

    def train(self,samples, responses):
        self.classifier.train(np.array(samples, dtype=np.float32),np.array(responses, dtype=np.float32))

    def predict(self, sample):
        return self.classifier.predict(np.matrix(sample))


    def save(self, url):
        print url
        self.classifier.save(url)
