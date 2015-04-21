import cv2
import numpy as np
from itertools import chain
import pickle
import bisect

class AbstractAlgorithm:
    algorithmName = 'Dense'
    def _initiateRegisters(self, frame):
        self.h, self.w = frame.shape[:2]
        self.distMat = np.zeros((480, 640))
        self.resultX, self.resultY =np.zeros(self.w), np.zeros(self.h)
        self.keyPointsAccumulator, self.keyPointsVector = [], []

    def getDistribution(self, i, m= 2):
        #distX = (np.arange(0,self.w)/float(self.w), self.rejectOutliers(self.resultX/float(i), m))
        #distY = (np.arange(0,self.h)/float(self.h), self.rejectOutliers(self.resultY/float(i), m))
        distX = (np.arange(0,self.w)/float(self.w), self.resultX/float(i))
        distY = (np.arange(0,self.h)/float(self.h), self.resultY/float(i))

        return distX, distY

    def analyze(self, frame, avaragePeriod):
        if avaragePeriod!=0:
            avarageMultiplier = 1 - 1/float(avaragePeriod)
        else:
            avarageMultiplier = 1
        tempResultX, tempResultY, self.processedFrame, tempResMat = self._getAnalysisResult(frame)
        self.distMat+= np.resize(tempResMat, self.distMat.shape)

        if len(self.keyPoints) > 0:
            self.keyPointsAccumulator.append(self.keyPoints/np.array([self.w, self.h]).astype(float))
            if len(self.keyPointsAccumulator)>= avaragePeriod:
                self.keyPointsVector = self.getCharacteristicPointsVector(self.keyPointsAccumulator)
                self.keyPointsAccumulator= []

        #self.resultX +=tempResultX
        self.resultX = self.resultX*avarageMultiplier+ tempResultX
        #self.resultY +=tempResultY
        self.resultY = self.resultY*avarageMultiplier+ tempResultY

    def rejectOutliers(self, dist, m = 2.):
        d = np.abs(dist - np.median(dist))
        mdev = np.median(d)
        thresh =  mdev+m*d
        place =np.place(dist, dist < thresh, 0.)
        return dist

    def getResultDict(self, i, m=2):
        resultDict ={}
        #distX, distY = self.getDistribution(i, m)
        #resultDict['distribution'] = {'x':distX, 'y':distY}
        #wCX, wCY = self.getWeightCenter(distX[1], distY[1])
        #resultDict['weightCenter'] = {'x':wCX, 'y':wCY}
        #resultDict['distMat'] = self.distMat
        #resultDict['keyPoints'] = list(chain(*self.keyPointsAccumulator))
        resultDict['keyPointsVector'] = self.keyPointsVector
        return resultDict

    def getWeightCenter(self, distX, distY):
        xSum, ySum, denominator = 0,0, 0
        lenDistX, lenDistY = len(distX), len(distY)
        for i in range(lenDistX):
            xSum += abs(distX[i])*i
            denominator +=abs(distX[i])
        if denominator >0:
            x= xSum/float(denominator)/ float(self.w)
        else:
            x=0
        denominator =0
        for i in range(lenDistY):
            ySum += abs(distY[i])*i
            denominator +=abs(distY[i])
        if denominator >0:
            y= ySum/float(denominator) / float(self.h)
        else:
            y =0
        return round(x,3), round(y, 3)

    # specyfic for each algorithm
    def _getAnalysisResult(self, frame):
        """ This function returns two np.arrays of image width and height size """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resX, resY = np.zeros(self.w), np.zeros(self.h)
        detector = cv2.FeatureDetector_create(self.algorithmName)
        pts = detector.detect(frame,None)
        distMat = np.zeros(frame.shape[:2])
        self.keyPoints = []
        for point in pts:
            resX[int(point.pt[0])]+=1
            resY[int(point.pt[1])]+=1
            cv2.circle(frame, (int(point.pt[0]), int(point.pt[1])),5,(255,255,255),1)
            distMat[point.pt[1]] [point.pt[0]]+=1
            self.keyPoints.append(tuple(point.pt))
        return resX, resY, frame, distMat

    """
    #  0 1 2 3 4 5 6 7
    #0| | | | | | | | |
    #1| | | | | | | | |
    #2| | | | | | | | |
    #3| | | | | | | | |
    #4| | | | | | | | |
    #5| | | | | | | | |
    #6| | | | | | | | |
    #7| | | | | | | | |
    """
    def getCharacteristicPointsVector(self, keyPtsList):
        matSideLen = 8
        #stepW = self.w/matSideLen/float(self.w)
        #stepH = self.h/matSideLen/float(self.h)
        stepW = 1/8.0
        stepH = 1/8.0

        #wRange = np.arange(stepW ,self.w+1, stepW)
        wRange = np.arange(stepW ,1, stepW)
        #hRange = np.arange(stepH ,self.h+1, stepH)
        hRange = np.arange(stepH ,1, stepH)
        #print wRange, hRange, keyPtsList
        charPtsDistVector = [0]*8*8
        for keyPt in keyPtsList[0]:
            wPos = bisect.bisect_left(wRange, keyPt[0])
            hPos = bisect.bisect_left(hRange, keyPt[1])
            #print keyPt, " at position ", (wPos, hPos)
            totPos = wPos + hPos*8
            #print wPos , hPos
            charPtsDistVector[totPos] += 1
        return charPtsDistVector