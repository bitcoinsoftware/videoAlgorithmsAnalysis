from PyQt4 import QtGui
import cv2, numpy as np

class Display:
    def __init__(self, qlabel, opencvPreview=False, secondaryDisplay=False):
        self.displayWidget = qlabel
        self.h, self.w = qlabel.height(), qlabel.width()
        self.opencvPreview = opencvPreview
        self.secondaryPreview = secondaryDisplay

    def display(self, frame, scaleToHeight = None):
        #frame = cv2.resize(frame, (self.w, self.h))
        image = QtGui.QImage(frame.tostring(), frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(image)
        """
        if self.secondaryPreview==True and self.opencvPreview==True: # if no raw preview requested
            cv2.imshow("raw preview ", frame)
        elif self.secondaryPreview==True:
            frame = np.zeros((100,100,3), dtype=np.uint8)
            cv2.imshow("Workaround ", frame)
        if 0xFF & cv2.waitKey(5) == 27:
             exit()
        """
        if scaleToHeight:
            pixmap = pixmap.scaledToHeight(self.h)
        else:
            pixmap = pixmap.scaledToWidth(self.w)
        self.displayWidget.setPixmap(pixmap)
