from AbstractAlgorithm import *

class Canny(AbstractAlgorithm):
    def _getAnalysisResult(self, frame):
        """ This function returns two np.arrays of image width and height size """
        resX, resY = np.zeros(self.w), np.zeros(self.h)
        x,y,step=0,0,4
        frame =cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.Canny(frame, 50, 200, 3)
        clpdFrame = np.clip(frame, 0,1)
        resX = np.sum(clpdFrame , axis=0)
        resY = np.sum(clpdFrame , axis=1)
        return resX, resY, frame, frame


