from AbstractAlgorithm import *

class OpticalFlow2DSlam(AbstractAlgorithm):
    def __init__(self):
        self.prevImg = None
        self.bigMap = None

    def _getAnalysisResult(self, frame):
        resX, resY = np.zeros(self.w), np.zeros(self.h) # just for compatibility with the gui
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[self.h/5:self.h/5*4, self.w/5:self.w/5*4]
        if self.prevImg!=None:
            pts= cv2.goodFeaturesToTrack(self.prevImg, self.h, 0.2, 5, useHarrisDetector=True)
            if pts !=None:
                nextPts, st, err = cv2.calcOpticalFlowPyrLK(self.prevImg, frame, pts, None)
                m=[0,0]
                ptsNumb = float(len(pts))
                for p0, pk in zip(pts, nextPts):
                    p0, pk = p0[0], pk[0]
                    m[0]+= pk[0]-p0[0]
                    m[1]+= pk[1]-p0[1]
                dX, dY =  int(m[0]/ptsNumb), int(m[1]/ptsNumb)
                tempMap = np.zeros((self.h+abs(dY), self.w+abs(dX) ,3), np.uint8)
                self.bigMap =tempMap
            else:
                #self.bigMap = frame
                pass
        else:
            #self.bigMap = frame
            pass

        self.prevImg= frame
        return resX, resY, frame
