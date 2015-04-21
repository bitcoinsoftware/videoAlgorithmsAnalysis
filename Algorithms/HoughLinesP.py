from AbstractAlgorithm import *
from math import cos, sin, pi 

class HoughLinesP(AbstractAlgorithm):
    def _getAnalysisResult(self, frame):
        """ This function returns two np.arrays of image width and height size """
        resX, resY = np.zeros(self.w), np.zeros(self.h)
        x,y=0,0
        frame =cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.Canny(frame, 50, 200, 3)
        distMat= np.zeros(frame.shape[:2])
        try:
            lines = cv2.HoughLinesP(frame,1, pi/180,50, minLineLength = 30, maxLineGap = 10)
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(frame,(x1,y1),(x2,y2),(255,255,255),3,8)
                resX[int((x1+x2)/2)]+=1
                resY[int((y1+y2)/2)]+=1
                distMat[x1][y1]+=1
        except:
            lines=[]
        return resX, resY , frame, distMat

