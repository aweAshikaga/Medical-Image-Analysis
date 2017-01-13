import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import cv2


class Model(object):
    def __init__(self):
        self.img = np.zeros((1, 1), np.uint8)

    def setImage(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        print(self.img )

    def contrast(self):
        rpy2.robjects.numpy2ri.activate()
        robjects.r['source']('R_functions.R')
        adjustContrast = robjects.r['adjustContrast']
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.img = clahe.apply(self.img)

        # self.img = cv2.equalizeHist(self.img)
        # self.img = np.array(adjustContrast(self.img / 255, 4))

    def getImage(self):
        return self.img
