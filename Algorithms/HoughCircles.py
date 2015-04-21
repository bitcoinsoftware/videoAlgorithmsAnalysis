from AbstractAlgorithm import *

class HoughCircles(AbstractAlgorithm):
    def _getAnalysisResult(self, frame):
        """ This function returns two np.arrays of image width and height size """
        resX, resY = np.zeros(self.w), np.zeros(self.h)
        x,y=0,0
        frame =cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pts = cv2.HoughCircles(frame, cv2.cv.CV_HOUGH_GRADIENT,0.5,10,param1=100,param2=30,minRadius=2,maxRadius=35)
        distMat= np.zeros(frame.shape[:2])
        if pts != None:
            pts = np.uint16(np.around(pts))
            for point in pts[0]:
                resX[int(point[0])]+=1
                resY[int(point[1])]+=1
                cv2.circle(frame,(point[0],point[1]),5,(255,255,255),1)
                distMat[point[1]] [point[0]]+=1
        return resX, resY, frame, distMat


