from AbstractAlgorithm import *
import time

class ImageChanges(AbstractAlgorithm):
    def __init__(self):
        self.prevImg = None
        self.prevCenter = (0,0)
        self.xSum, self.ySum, self.k = 0,0,0
        self.upDownHistory =[]
 
    def _findHoleCenter(self, frameCopy):
        frameCopy = cv2.absdiff(frameCopy, self.prevImg)
        ret, frameCopy = cv2.threshold(frameCopy,2,255,cv2.THRESH_BINARY)
        frameCopy = cv2.GaussianBlur(frameCopy, (15,15), 5)
        ret, frameCopy = cv2.threshold(frameCopy,40,255,cv2.THRESH_BINARY)
        resX = np.sum(frameCopy , axis=0)
        resY = np.sum(frameCopy , axis=1)
        x, y = self.getWeightCenter(resX, resY)
        if x ==0 or y ==0:
            x, y = self.xSum/self.k, self.ySum/self.k
        self.xSum+=x
        self.ySum+=y
        self.k+=1
        cenX, cenY = int(self.xSum/self.k), int(self.ySum/self.k)
        upPoint, downPoint = self._findHoleYBounds(frameCopy, (cenX, cenY))

        return resX, resY, (cenX, cenY), upPoint, downPoint

    #TODO
    def _findHoleYBounds(self, frame, centerPoint):
        fast = cv2.FastFeatureDetector()
        pts = fast.detect(frame,None)
        upY, downY = [], []
        for point in pts:
            if point.pt[1]<centerPoint[1]:
                upY.append(point.pt[1])
            else: 
                downY.append(point.pt[1])
        upY.sort()
        downY.sort()
        upY, downY = upY[:5], downY[-5:]
        if upY:
            upCenter = (centerPoint[0], int(np.mean(upY)) )
        else:
            upCenter = None
        if downY:
            downCenter = (centerPoint[0], int(np.mean(downY)))
        else:
            downCenter = None
        """
        #counting optical flow
        upOF, downOf = 0, 0
        if self.prevImg!=None:
            upY, downY = [], []
            for point in pts:
                if point.pt[1]<centerPoint[1]:
                    upY.append(point.pt)
                else:
                    downY.append(point.pt)
            if len(upY)>0:
                #nextPts, st, err = cv2.calcOpticalFlowPyrLK( frame,self.prevImg, np.array(upY) , None)
                upOF = cv2.calcOpticalFlowFarneback(self.prevImg, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                print upOF
                #for nPt, oPt in zip(nextPts,upY):
                #    upOF += oPt[1]-nPt[1]
        self.prevImg = frame
        """
        return upCenter, downCenter#, upOF

    def _getAnalysisResult(self, frame):
        frameCopy = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resX, resY = np.zeros(self.w), np.zeros(self.h)
        if self.prevImg!=None and self.prevImg.shape == frameCopy.shape:
            resX, rexY, holeCenter, upPoint, downPoint = self._findHoleCenter(frameCopy)
            dupa = frame.copy()
            cv2.circle(dupa , holeCenter, 5, (255,255,255), thickness=3)
            if upPoint:
                cv2.circle(dupa, upPoint, 5, (255,255,255), thickness=3)
            if downPoint:
                cv2.circle(dupa, downPoint, 5, (255,255,255), thickness=3)
            if upPoint and downPoint:
                #self.upDownHistory.append(downPoint[0]-upPoint[0])
                print downPoint[1]-upPoint[1], time.time()
                #print upOF, time.time()

            self.prevImg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return resX, resY, dupa,frameCopy
        else:
            self.prevImg= frameCopy
            return resX, resY, frameCopy, frameCopy
