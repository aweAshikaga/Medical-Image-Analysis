import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import cv2


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

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)

class Image(Observable):
    def __init__(self):
        Observable.__init__(self)
        self.img = np.zeros((1, 1), np.uint8)
        self._zoomFactor = 1

    @property
    def zoomFactor(self):
        return self._zoomFactor

    @zoomFactor.setter
    def zoomFactor(self, newValue):
        if newValue > 0:
            self._zoomFactor = newValue
            self.update_observers()
        else:
            self._zoomFactor = 0.25

    def openFile(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self._zoomFactor = 1
        self.update_observers()

    def addContrast(self):
        rpy2.robjects.numpy2ri.activate()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        self.img = clahe.apply(self.img)
        self.update_observers()

    def addContrast2(self, value):
        self.img = cv2.multiply(self.img,value)
        self.update_observers()
        print(self.img)

    def getImageDimensions(self):
        return self.img.shape

    def getImage(self, zoomFactor=1):
        if self._zoomFactor == 1:
            return self.img
        elif self._zoomFactor > 1:
            return cv2.resize(self.img,None,fx=self._zoomFactor,fy=self._zoomFactor, interpolation = cv2.INTER_CUBIC)
        elif 0 < self._zoomFactor < 1:
            return cv2.resize(self.img, None, fx=self._zoomFactor, fy=self._zoomFactor, interpolation=cv2.INTER_AREA)
