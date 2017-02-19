import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import cv2
import math
import collections
import model.algorithms as algorithms


class Observable(object):
    def __init__(self):
        self.observers = []

    def register(self, observer):
        if not observer in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, sender, *args, **kwargs):
        for observer in self.observers:
            observer.update(sender, *args, **kwargs)


class History(object):
    """ The History class provides the possibility to keep track of changes that were done
        to a variable and move backwards and forward between these changes.
    """

    def __init__(self):
        # undo-stack
        self.undo = collections.deque()
        # redo-stack
        self.redo = collections.deque()

    def goBackwards(self, currentValue):
        """ Returns and pops the last saved value of the undo-stack and stores the current value on the redo-stack.
        """
        if len(self.undo) >= 1:
            self.redo.append(currentValue)
            return self.undo.pop()
        else:
            return currentValue

    def goForward(self, currentValue):
        """ Returns and pops the last value of the redo-stack and stores the current value one the undo-stack.
        """
        if len(self.redo) > 0:
            self.undo.append(currentValue)
            return self.redo.pop()
        else:
            return currentValue


class Image(Observable):
    def __init__(self):
        Observable.__init__(self)
        self._img = None
        self._zoomFactor = 1
        self.imgHistory = History()

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        if self.img is not None:
            self.imgHistory.undo.append(self.img)

        self._img = value

    @property
    def zoomFactor(self):
        return self._zoomFactor

    @zoomFactor.setter
    def zoomFactor(self, newValue):
        if newValue > 0:
            if newValue <= 3:
                self._zoomFactor = newValue
            else:
                self._zoomFactor = 3
        else:
            self._zoomFactor = 0.25

        self.update_observers(self)

    def undo(self):
        self._img = self.imgHistory.goBackwards(self.img)
        self.update_observers(self)

    def redo(self):
        self._img = self.imgHistory.goForward(self.img)
        self.update_observers(self)

    def openFile(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self._zoomFactor = 1
        self.update_observers(self)

    def saveImage(self, path):
        if self.img is not None:
            cv2.imwrite(path, self.img)

    def addContrastCLAHE(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        self.img = clahe.apply(self.img)
        self.update_observers(self)
        self.imgHistory.redo.clear()

    def addContrastCustom(self, value):
        self.img = cv2.multiply(self.img, value)
        self.update_observers(self)
        self.imgHistory.redo.clear()

    def getImageDimensions(self):
        if self.img is not None:
            return self.img.shape
        else:
            return 0, 0

    def getZoomedImage(self):
        if self._zoomFactor == 1:
            return self.img
        elif self._zoomFactor > 1:
            return cv2.resize(self.img,None,fx=self._zoomFactor,fy=self._zoomFactor, interpolation=cv2.INTER_CUBIC)
        elif 0 < self._zoomFactor < 1:
            return cv2.resize(self.img, None, fx=self._zoomFactor, fy=self._zoomFactor, interpolation=cv2.INTER_AREA)

    def filterSmoothingAverage(self, kernelSize):
        if self.img is not None:
            self.img = cv2.blur(self.img, kernelSize)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def filterSmoothingMedian(self, kernelSize):
        if self.img is not None:
            self.img = cv2.medianBlur(self.img, kernelSize)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def filterSmoothingGaussian(self, kernelSize):
        if self.img is not None:
            self.img = cv2.GaussianBlur(self.img, kernelSize, 0)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def derivativeLaPlace(self):
        if self.img is not None:
            self.img = cv2.Laplacian(self.img, cv2.CV_8U)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def derivativeSobel(self):
        if self.img is not None:
            self.img = cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=5)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def thresholdManual(self, threshold):
        if self.img is not None:
            self.img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)[1]
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def thresholdAdaptive(self):
        if self.img is not None:
            self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 2)
            #self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 2)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def thresholdOtsu(self):
        if self.img is not None:
            self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def sharpen(self):
        if self.img is not None:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.img = cv2.filter2D(self.img, -1, kernel)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def edges(self):
        if self.img is not None:
            self.img = cv2.Canny(self.img, 50, 150, apertureSize=3)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def hughLinesP(self):
        if self.img is not None:
            #edges = cv2.Canny(self.img, 50, 150, apertureSize=3)
            #self.img = (255-edges)
            #edges = (255-edges)
            edges = self.img

            minLineLength =  100
            maxLineGap = 100
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 255, minLineLength, maxLineGap)

            print(lines[0])
            for x in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    cv2.line(self.img, (x1, y1), (x2, y2), 127, 1)

            self.update_observers(self)
            self.imgHistory.redo.clear()

    def hughLines(self):
        if self.img is not None:
            edges = cv2.Canny(self.img, 50, 150, apertureSize=3)
            #self.img = cv2.Canny(self.img, 50, 150, apertureSize=3)
            #edges = (255 - edges)

            #edges = self.img

            angles = []

            lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
            for x in range(0, len(lines)):
                for rho, theta in lines[x]:
                    angles.append(theta * (180/math.pi))
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 5000 * (-b))
                    y1 = int(y0 + 5000 * (a))
                    x2 = int(x0 - 5000 * (-b))
                    y2 = int(y0 - 5000 * (a))

                    cv2.line(self.img, (x1, y1), (x2, y2), 127, 2)

            self.update_observers(self)
            self.imgHistory.redo.clear()
            return angles

    #FindCountour
    #FindOrientation of contour

    def watershedSegmentation(self):
        if self.img is not None:


            ret, thresh = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            # convert single channel to three channels
            temp = np.empty((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
            temp[:, :, 0] = self.img
            temp[:, :, 1] = self.img
            temp[:, :, 2] = self.img

            # Problem: cv2.watershed expects 3 channel input. self.img has to be converted first.
            markers = cv2.watershed(temp, markers)
            temp[markers == -1] = 255

            self.img = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def contours(self):
        if self.img is not None:
            ret, thresh = cv2.threshold(self.img, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnt = contours[0]
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            print(angle)

    def topHatTransformation(self):
        if self.img is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.img = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, kernel)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def dilation(self):
        if self.img is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.img = cv2.dilate(self.img, kernel, iterations=1)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def skeletonization(self):
        if self.img is not None:
            self.img = algorithms.skeletonization(self.img)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def getPorosity(self):
        if self.img is not None:
            blackPixelCount = (self.img == 0).sum()
            totalPixelCount = (self.img.shape[0] * self.img.shape[1])
            porosity = blackPixelCount / totalPixelCount
            porosityPercentage = porosity * 100
            print("Porosity: {0:.2f}%".format(porosityPercentage))
            return porosity

