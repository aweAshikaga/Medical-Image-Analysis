import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import cv2
import collections


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