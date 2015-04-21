from AbstractAlgorithm import *

class OpticalFlow(AbstractAlgorithm):
    def __init__(self):
        self.prevImg = None

    def _getAnalysisResult(self, frame):
        """ This function returns two np.arrays of image width and height size """
        resX, resY = np.zeros(self.w), np.zeros(self.h)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        distMat= np.zeros(frame.shape[:2])
        if self.prevImg!=None and self.prevImg.shape == frame.shape:
            pts= cv2.goodFeaturesToTrack(frame, self.h+self.w, 0.1, 2, useHarrisDetector=True)
            print pts
            frameCopy = np.copy(frame)
            if pts!=None:
                nextPts, st, err = cv2.calcOpticalFlowPyrLK(self.prevImg, frame, pts, None)

                for prevPt, newPt in zip(pts, nextPts):
                    resX[prevPt[0][0]]+=prevPt[0][0]-newPt[0][0]
                    resY[prevPt[0][1]]+=prevPt[0][1]-newPt[0][1]
                    cv2.circle(frameCopy, (int(prevPt[0][0]), int(prevPt[0][1])),5,(0,0,255),1)
                    cv2.circle(frameCopy, (int(newPt[0][0]), int(newPt[0][1])),5,(0,255,0),1)
                    distMat[int(newPt[0][1])] [int(newPt[0][0])]+=1
            self.prevImg= frame
            return resX, resY, frameCopy,distMat
        else:
            self.prevImg= frame
            return resX, resY, frame, distMat

