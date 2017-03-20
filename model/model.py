import numpy as np
import cv2
import math
import collections
import skimage.morphology
import skimage.transform
import time # For performance testing. Should be deleted later


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
    def __init__(self, imgData = None, scale = 0):
        Observable.__init__(self)

        if imgData is not None:
            self._img = np.array(imgData, np.uint8)
        else:
            self._img = None

        self._zoomFactor = 1
        self.scale = scale
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

    def isBinary(self):
        """ Returns True if the image is binary in black and white (values 0 or 255).
        """
        if self.img is not None:
            isBinary = np.all(np.logical_or(self.img == 0, self.img == 255))
            if isBinary:
                return True
            else:
                return False

    def addContrastCLAHE(self):
        if self.img is not None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
            self.img = clahe.apply(self.img)
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def addContrastCustom(self, value):
        if self.img is not None:
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
            return cv2.resize(self.img,None,fx=self._zoomFactor, fy=self._zoomFactor, interpolation=cv2.INTER_CUBIC)
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

    def houghLines(self):
        if self.img is not None:
            #edges = cv2.Canny(self.img, 50, 150, apertureSize=3)
            edges = self.img
            angles = []

            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=300)
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

    def houghLines2(self, threshold, maxLines=np.inf):
        if self.img is not None:
            out, angles2, d = skimage.transform.hough_line(self._img)

            out, angles2, d = skimage.transform.hough_line_peaks(out, angles2, d, threshold=threshold*np.amax(out), num_peaks=maxLines)

            angles = []
            for rho, theta in zip(d, angles2):
                print(rho)
                print(np.rad2deg(theta))
                print("----")
                angles.append(theta * (180 / math.pi))
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 5000 * (-b))
                y1 = int(y0 + 5000 * a)
                x2 = int(x0 - 5000 * (-b))
                y2 = int(y0 - 5000 * a)

                cv2.line(self._img, (x1, y1), (x2, y2), 127, 5)

            self.update_observers(self)
            self.imgHistory.redo.clear()
            return angles

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
        """ Creates a skeleton of the image with the implementation of Zhang Suen.
            The Image has to be binary.
        """
        if self.img is not None:
            #self.img = self.img

            # Convert from OpenCV format to scikit-image format.
            self.convertFromOpenCVtoScikitImage()

            # Use scikit-image skeletonize implementation (Zhang Suen algorithm)
            self._img = skimage.morphology.skeletonize(self._img)

            # Convert format back from scikit-image to OpenCV
            self.convertFromScikitImageToOpenCV()

            # Update
            self.update_observers(self)
            self.imgHistory.redo.clear()

    def convertFromOpenCVtoScikitImage(self):
        self._img[self._img == 255] = 1

    def convertFromScikitImageToOpenCV(self):
        self._img = self._img.astype(np.uint8)
        self._img[self._img == 1.0] = 255
        return self._img

    def getPorosity(self):
        """ Returns the ratio of black pixels to the overall pixel count.
        """
        if self.img is not None:
            blackPixelCount = (self.img == 0).sum()
            totalPixelCount = (self.img.shape[0] * self.img.shape[1])
            porosity = blackPixelCount / totalPixelCount
            return porosity

    def getDiameters(self):
        """ Returns a numpy array with all measured diameters.
        """

        start_time = time.time() # For performance measure. Can be deleted later
        diameters = np.array([]) # return value. Array of all measured diameters.

        # Find all rows with a least one white pixel in it
        relevantRows = np.unique(np.where(self.img == 255)[0])

        for currentRow in relevantRows:
            # Extract the pixel values of the current row.
            row = self.img[currentRow, :]

            rowHasOnlyWhiteValues = np.all(row == 255)

            # Create an array which stores the indices of all white pixels of the current row.
            indicesWithWhiteValue = np.where(row == 255)[0]

            # Create a list to store tuples of start index and distance of consecutive white pixels.
            distances = []

            # Initialize a variable which will store the distance of one section of white pixels.
            distance = 1

            # Check if there is at least one white pixel in the current row.
            if len(indicesWithWhiteValue) > 0:
                start_index = indicesWithWhiteValue[0]
            else:
                start_index = 0

            # Find the distance of each section of white pixels in horizontal direction.
            if len(indicesWithWhiteValue) == 1:
                # Trivial case: Only one white pixel: Save the index of that white pixel with a length of one.
                distances.append((start_index, distance))
            elif len(indicesWithWhiteValue) > 1:
                # remember the index of the previously checked white pixel.
                previousIndex = indicesWithWhiteValue[0]

                for index in indicesWithWhiteValue[1:]:
                    if previousIndex + 1 == index:
                        distance += 1

                        # When the last pixel is reached, append the distance nevertheless.
                        if index == indicesWithWhiteValue[-1]:
                            distances.append((start_index, distance))
                    else:
                        distances.append((start_index, distance))
                        distance = 1
                        start_index = index

                    previousIndex = index

            # Find the distances in vertical direction.
            for start, distance in distances:
                column = self.img[:, int(start + distance / 2)]
                length = 0

                reachedEdgeBottom = False
                reachedEdgeTop = False

                blackValues = np.where(column[currentRow:] == 0)[0]
                if blackValues.size > 0:
                    length = min(blackValues)
                else:
                    reachedEdgeBottom = True

                if reachedEdgeBottom or rowHasOnlyWhiteValues:
                    lengthToBottom = length
                    blackValues = np.where(column[0:currentRow] == 0)[0]
                    print(blackValues)
                    if blackValues.size > 0:
                        length = currentRow - max(blackValues)
                        print(length)
                    else:
                        reachedEdgeTop = True

                if reachedEdgeTop and not rowHasOnlyWhiteValues:
                    diameters = np.append(diameters, distance)
                elif rowHasOnlyWhiteValues and not reachedEdgeTop:
                    diameters = np.append(diameters, length + lengthToBottom - 1)
                elif reachedEdgeTop and rowHasOnlyWhiteValues:
                    # if there are only white values in horizontal and vertical direction,
                    # there cannot determine the actual diameter of the fiber. Therefore pass
                    pass
                else:
                    alpha = np.arctan(distance / (2 * length))
                    d = distance * np.cos(alpha)
                    diameters = np.append(diameters, d)

        print("Execution time: " + str(time.time() - start_time))
        return diameters
