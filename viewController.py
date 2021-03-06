import view.Ui as Ui
import model.model as model
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication, QPainter, QColor
from PyQt5.QtCore import Qt

class ViewController(object):
    def __init__(self):
        super().__init__()
        self.mainWindow = Ui.MainWindow()
        self.connectActions()

        self._isDefiningArea = False
        self._isDefiningScale = False
        self._initialAreaPoint = (-1, -1)
        self._endAreaPoint = (-1, -1)

    @property
    def currentImageObject(self):
        return self.mainWindow.tabBar.tabData(self.mainWindow.tabBar.currentIndex())

    def connectActions(self):
        """Connect the view action signals to their respective slots
        """

        self.mainWindow.actionOpen.triggered.connect(self.openFile)
        self.mainWindow.actionContrast.triggered.connect(self.addContrast)
        self.mainWindow.actionZoomIn.triggered.connect(self.zoomIn)
        self.mainWindow.actionZoomOut.triggered.connect(self.zoomOut)
        self.mainWindow.actionReset_zoom.triggered.connect(self.resetZoom)
        self.mainWindow.actionRedo.triggered.connect(self.redo)
        self.mainWindow.actionUndo.triggered.connect(self.undo)
        self.mainWindow.tabBar.currentChanged.connect(self.update)
        self.mainWindow.actionSmoothingFilter.triggered.connect(self.addSmoothing)
        self.mainWindow.actionDerivativeLaPlace.triggered.connect(self.addDerivativeLaPlace)
        self.mainWindow.actionDerivativeSobel.triggered.connect(self.addDerivateSobel)
        self.mainWindow.actionSave.triggered.connect(self.saveImage)
        self.mainWindow.actionWatershed_Transformation.triggered.connect(self.addWatershedSegmentation)
        self.mainWindow.actionSegmentation.triggered.connect(self.addSegmentation)
        self.mainWindow.actionSharpen.triggered.connect(self.addSharpening)
        self.mainWindow.actionFiberOrientation.triggered.connect(self.addHoughLines)
        self.mainWindow.actionTop_Hat_Transformation.triggered.connect(self.addTopHatTransformation)
        self.mainWindow.actionDilation.triggered.connect(self.addDilation)
        self.mainWindow.actionSkeletonization.triggered.connect(self.addSkeletonization)
        self.mainWindow.actionPorosity.triggered.connect(self.checkPorosity)
        self.mainWindow.actionEdgeDetection.triggered.connect(self.addEdges)
        self.mainWindow.actionDefineAreas.triggered.connect(self.grabArea)
        self.mainWindow.actionDefineScale.triggered.connect(self.grabScale)
        self.mainWindow.actionFiberDiameter.triggered.connect(self.findDiameters)
        self.mainWindow.actionExport_diameters.triggered.connect(self.exportDiametersToCSV)
        self.mainWindow.actionExport_orientation_angles.triggered.connect(self.exportAnglesToCSV)
        self.mainWindow.lblImgDisplay.setMouseTracking(True)
        self.mainWindow.lblImgDisplay.mousePressEvent = self.getMousePositionOnLabel
        self.mainWindow.lblImgDisplay.mouseReleaseEvent = self.getMousePositionOnLabel
        self.mainWindow.lblImgDisplay.mouseMoveEvent = self.getMousePositionOnLabel
        self.mainWindow.keyPressEvent = self.keyPressEvent

    def checkIfBinary(self):
        if self.currentImageObject:
            isBinary = self.currentImageObject.isBinary()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelDefineAreaOrScale()

    def cancelDefineAreaOrScale(self):
        self.mainWindow.menubar.setEnabled(True)
        QGuiApplication.setOverrideCursor(Qt.ArrowCursor)
        self._isDefiningArea = False
        self._isDefiningScale = False
        self._initialAreaPoint = (-1, -1)
        self._endAreaPoint = (-1, -1)
        self.displayImage()

    def drawRectOnLabel(self, point1, point2):
        if self._isDefiningArea and self.currentImageObject and self._initialAreaPoint != (-1, -1):
            self.displayImage()
            qp = QPainter()
            qp.begin(self.mainWindow.lblImgDisplay.pixmap())
            qp.setPen(QColor(255, 0, 0))
            x = point1[0]
            y = point1[1]
            width = point2[0] - point1[0]
            height = point2[1] - point1[1]
            qp.drawRect(x, y, width, height)
            qp.end()
            self.mainWindow.lblImgDisplay.repaint()

    def drawLineOnLabel(self, point1, point2):
        if self._isDefiningScale and self.currentImageObject and self._initialAreaPoint != (-1, -1):
            self.displayImage()
            qp = QPainter()
            qp.begin(self.mainWindow.lblImgDisplay.pixmap())
            qp.setPen(QColor(255, 0, 0))
            x1 = point1[0]
            y1 = point1[1]
            x2 = point2[0]
            y2 = point2[1]
            qp.drawLine(x1, y1, x2, y2)
            qp.end()
            self.mainWindow.lblImgDisplay.repaint()

    def grabArea(self):
        self.mainWindow.menubar.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CrossCursor)
        self._isDefiningArea = True

    def grabScale(self):
        self.mainWindow.menubar.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CrossCursor)
        self._isDefiningScale = True

    def getMousePositionOnLabel(self, event):
        if self._isDefiningArea or self._isDefiningScale:
            x = event.pos().x()
            y = event.pos().y()

            pixmapInitialAreaPoint = self.convertLabelPointToPixmapPoint(self._initialAreaPoint)
            pixmapEndAreaPoint = self.convertLabelPointToPixmapPoint(self._endAreaPoint)

            if self._initialAreaPoint == (-1, -1) and event.button() == 1:
                # Left mouse button was pressed for the first time while defining area.
                self._initialAreaPoint = (x, y)
            elif self._initialAreaPoint != (-1, -1) and event.button() == 0:
                # The rect angle is currently drawing and no button has been pressed.
                self._endAreaPoint = (x, y)

                if self._isDefiningArea:
                    self .drawRectOnLabel(pixmapInitialAreaPoint, pixmapEndAreaPoint)
                elif self._isDefiningScale:
                    self.drawLineOnLabel(pixmapInitialAreaPoint, pixmapEndAreaPoint)
            elif event.button() == 2:
                # Right mouse button has been pressed to cancel the process.
                self.cancelDefineAreaOrScale()
            elif self._initialAreaPoint != (-1, -1) and event.button() == 1:
                # Left mouse button was released
                self.mainWindow.menubar.setEnabled(True)
                QGuiApplication.setOverrideCursor(Qt.ArrowCursor)

                if self._isDefiningScale:
                    self.drawLineOnLabel(pixmapInitialAreaPoint, pixmapEndAreaPoint)
                    self._isDefiningScale = False
                    self.mainWindow.lblImgDisplay.repaint()

                    um, ok = QInputDialog.getInt(self.mainWindow, "Scale", "um: ", 100)

                    if ok:
                        initPoint = self.convertZoomedPointToOriginalPoint(pixmapInitialAreaPoint)
                        endPoint = self.convertZoomedPointToOriginalPoint(pixmapEndAreaPoint)
                        startX = int(initPoint[0])
                        startY = int(initPoint[1])
                        endX = int(endPoint[0])
                        endY = int(endPoint[1])

                        if startX > endX:
                            startX, endX = endX, startX

                        if startY > endY:
                            startY, endY = endY, startY

                        distance = math.sqrt(math.pow(endX-startX, 2) + math.pow(endY-startY, 2))
                        scale = um / distance

                        self.currentImageObject.scale = scale

                        # Update the image, so that it shows on screen.
                        self.displayImage()

                        # Set the coordinates responsible for defining the area back to their basic state.
                        self._initialAreaPoint = (-1, -1)
                        self._endAreaPoint = (-1, -1)

                        self.update()

                    self._initialAreaPoint = (-1, -1)
                    self._endAreaPoint = (-1, -1)

                    self.update()

                if self._isDefiningArea:
                    self.drawRectOnLabel(pixmapInitialAreaPoint, pixmapEndAreaPoint)
                    self._isDefiningArea = False
                    self.mainWindow.lblImgDisplay.repaint()

                    # Create new tab with defined area
                    initPoint = self.convertZoomedPointToOriginalPoint(pixmapInitialAreaPoint)
                    endPoint = self.convertZoomedPointToOriginalPoint(pixmapEndAreaPoint)
                    startX = int(initPoint[0])
                    startY = int(initPoint[1])
                    endX = int(endPoint[0])
                    endY = int(endPoint[1])

                    if startX > endX:
                        startX, endX = endX, startX

                    if startY > endY:
                        startY, endY = endY, startY

                    imageDataOfArea = self.currentImageObject.img[startY:endY, startX:endX]
                    scaleOfCurrentImage = self.currentImageObject.scale

                    # Add a new tab (with filename as caption) and remember its index.
                    newTabBarIndex = self.mainWindow.tabBar.addTab("new")

                    # Make the new tab the active tab.
                    self.mainWindow.tabBar.setCurrentIndex(newTabBarIndex)

                    # Create a new image object for the new tab.
                    newImage = model.Image(imageDataOfArea, scaleOfCurrentImage)

                    # Register this viewController as an observer to the image object.
                    newImage.register(self)

                    # Save the reference to the image object in the tab data of the newly created tab.
                    self.mainWindow.tabBar.setTabData(newTabBarIndex, newImage)

                    # Update the image, so that it shows on screen.
                    self.displayImage()

                    # Set the coordinates responsible for defining the area back to their basic state.
                    self._initialAreaPoint = (-1, -1)
                    self._endAreaPoint = (-1, -1)

                    self.update()

    def convertZoomedPointToOriginalPoint(self, zoomedPoint):
        zoomFactor = self.currentImageObject.zoomFactor
        zoomedX = zoomedPoint[0]
        zoomedY = zoomedPoint[1]
        imgWidth = self.currentImageObject.img.shape[1]
        imgHeight = self.currentImageObject.img.shape[0]

        originalX = zoomedX / zoomFactor
        originalY = zoomedY / zoomFactor

        if originalX < 0:
            originalX = 0
        elif originalX > imgWidth:
            originalX = imgWidth

        if originalY < 0:
            originalY = 0
        elif originalY > imgHeight:
            originalY = imgHeight

        return originalX, originalY

    def convertLabelPointToPixmapPoint(self, labelPoint):
        labelX = labelPoint[0]
        labelY = labelPoint[1]

        labelWidth = self.mainWindow.lblImgDisplay.size().width()
        labelHeight = self.mainWindow.lblImgDisplay.size().height()
        pixmapWidth = self.mainWindow.lblImgDisplay.pixmap().size().width()
        pixmapHeight = self.mainWindow.lblImgDisplay.pixmap().size().height()

        offsetX = (labelWidth - pixmapWidth) / 2
        offsetY = (labelHeight - pixmapHeight) / 2

        pixmapX = labelX - offsetX
        pixmapY = labelY - offsetY

        if pixmapX < 0:
            pixmapX = 0
        elif pixmapX > pixmapWidth:
            pixmapX = pixmapWidth

        if pixmapY < 0:
            pixmapY = 0
        elif pixmapY > pixmapHeight:
            pixmapY = pixmapHeight

        return pixmapX, pixmapY

    def checkPorosity(self):
        """ Get the porosity of the current image.
        """
        if self.currentImageObject:
            porosity = self.currentImageObject.getPorosity() * 100
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setWindowTitle("Python Fiber Image Analyzer")
            msgbox.setText("The porosity is: {0:.2f} %".format(porosity))
            msgbox.exec_()

    def addSkeletonization(self):
        """ Add skeletonization to the current image.
        """
        if self.currentImageObject:
            if not self.currentImageObject.isBinary():
                msgbox = QMessageBox()
                msgbox.setIcon(QMessageBox.Information)
                msgbox.setWindowTitle("Python Fiber Image Analyzer")
                msgbox.setText("The image must be binary to perform diameter analysis.")
                msgbox.exec_()
            else:
                self.currentImageObject.skeletonization()

    def addTopHatTransformation(self):
        """ Add top hat transformation to the current image.
        """
        if self.currentImageObject:
            self.currentImageObject.topHatTransformation()

    def addDilation(self):
        """ Add dilation to the current image.
        """
        if self.currentImageObject:
            self.currentImageObject.dilation()

    def addEdges(self):
        if self.currentImageObject:
            self.currentImageObject.edges()

    def redo(self):
        if self.currentImageObject:
            self.currentImageObject.redo()

    def undo(self):
        if self.currentImageObject:
            self.currentImageObject.undo()

    def addHoughLines(self):
        if self.currentImageObject:
            if not self.currentImageObject.isBinary():
                msgbox = QMessageBox()
                msgbox.setIcon(QMessageBox.Information)
                msgbox.setWindowTitle("Python Fiber Image Analyzer")
                msgbox.setText("The image must be binary to perform orientation analysis.")
                msgbox.exec_()
            else:
                dialog = Ui.OrientationDialog()
                dialog.exec_()

                if dialog.result() == 1:
                    threshold = dialog.doubleSpinBoxThreshold.value()
                    maxLines = dialog.spinBoxMaxLines.value()

                    if maxLines == 0:
                        maxLines = np.inf

                    angles = self.currentImageObject.houghLines2(threshold, maxLines)

                    plt.style.use("ggplot")
                    #plt.hist(angles, bins=60, range=(- 90, 90), rwidth=0.8)
                    plt.hist(angles, bins=180, range=(0, 180), rwidth=0.8)
                    plt.xlabel("Orientation in degrees")
                    plt.ylabel("Frequency")
                    #plt.xticks([x for x in range(-90, 90+1, 30)])
                    plt.xticks([x for x in range(0, 180+1, 30)])
                    plt.title("Fiber Orientation")
                    plt.show()

    def findDiameters(self):
        if self.currentImageObject:
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setWindowTitle("Python Fiber Image Analyzer")
            if not self.currentImageObject.isBinary():
                msgbox.setText("The image must be binary to perform diameter analysis.")
                msgbox.exec_()
            else:
                minDiameter, ok = QInputDialog.getDouble(self.mainWindow, 'Diameter Analysis', 'Ignore all diameters smaller than (in %s):' % (self.currentImageObject.getScaleUnit()))

                if not ok:
                    return 0

                diameter = self.currentImageObject.getDiameters(min_diameter=minDiameter)

                if len(diameter) == 0:
                    msgbox.setText("No diameters found with the given parameters.")
                    msgbox.exec_()
                    return 0

                # Calc median:
                median_value = np.median(diameter)
                firstQuartile = np.percentile(diameter, 25)
                thirdQuartile = np.percentile(diameter, 75)

                a = plt.hist(diameter, bins=int(max(diameter))+1, range=(0, int(max(diameter))+1))
                maximum = max(a[0])

                # Find the first value that is equal or bigger than 1% of the maximum to use it as x-axis limit.
                y_x_values = list(zip(a[0], a[1]))
                for y, x in reversed(y_x_values):
                    if y >= 0.01 * maximum:
                        right_value = x
                        break

                unit = self.currentImageObject.getScaleUnit()
                plt.clf()
                plt.style.use("ggplot")
                plt.figure(1)
                plt.subplot(211)

                plt.hist(diameter, bins='doane', range=(0, right_value+1), rwidth=0.8)

                plt.xlabel("Fiber diameter in " + unit)
                plt.ylabel("Frequency")
                plt.title("Fiber diameters")

                plt.subplot(212)
                bp = plt.boxplot(diameter, patch_artist=True, labels=[''], showfliers=False)
                for median in bp["medians"]:
                    median.set(color="#000000", linewidth=1.0, )

                plt.ylabel("Fiber diameter in " + unit)
                plt.title(r"Q1: %.2f %s, $\tilde{x}$: %.2f %s, Q3: %.2f %s" % (firstQuartile, unit, median_value, unit, thirdQuartile, unit))

                plt.tight_layout()
                plt.show()

    def addContrast(self):
        if self.currentImageObject:
            dialog = Ui.ContrastDialog()
            dialog.exec_()

            if dialog.result() == 1:
                if dialog.radioBtnCLAHE.isChecked():
                    self.currentImageObject.addContrastCLAHE()
                elif dialog.radioBtnManual.isChecked():
                    contrastValue = dialog.doubleSpinBox.value()
                    self.currentImageObject.addContrastCustom(contrastValue)

    def addSmoothing(self):
        if self.currentImageObject:
            dialog = Ui.SmoothingDialog()
            dialog.exec_()
            if dialog.result() == 1:
                kernelSize = dialog.spinBoxKernelSize.value()
                if dialog.radioBtnAverage.isChecked():
                    self.currentImageObject.filterSmoothingAverage((kernelSize, kernelSize))
                elif dialog.radioBtnGaussian.isChecked():
                    self.currentImageObject.filterSmoothingGaussian((kernelSize, kernelSize))
                elif dialog.radioBtnMedian.isChecked():
                    self.currentImageObject.filterSmoothingMedian(kernelSize)

    def addSegmentation(self):
        if self.currentImageObject:
            dialog = Ui.SegmentationDialog()
            dialog.exec_()
            if dialog.result() == 1:
                if dialog.radioBtnManual.isChecked():
                    thresh = dialog.hSliderManual.value()
                    self.currentImageObject.thresholdManual(thresh)
                elif dialog.radioBtnAdaptive.isChecked():
                    self.currentImageObject.thresholdAdaptive()
                elif dialog.radioBtnOtsu.isChecked():
                    self.currentImageObject.thresholdOtsu()

    def addDerivativeLaPlace(self):
        if self.currentImageObject:
            self.currentImageObject.derivativeLaPlace()

    def addDerivateSobel(self):
        if self.currentImageObject:
            self.currentImageObject.derivativeSobel()

    def openFile(self):
        """ Open a file dialog and open the chosen file path.
        """
        filePath = QFileDialog.getOpenFileName(self.mainWindow, 'Open file', filter='*.jpg; *.png; *.tiff; *.tif')[0]

        if filePath:
            # Get the filename from the whole file path
            filename = os.path.split(filePath)[1]

            # Add a new tab (with filename as caption) and remember its index.
            newTabBarIndex = self.mainWindow.tabBar.addTab(filename)

            # Make the new tab the active tab.
            self.mainWindow.tabBar.setCurrentIndex(newTabBarIndex)

            # Create a new image object for the new tab.
            newImage = model.Image()

            # Register this viewController as an observer to the image object.
            newImage.register(self)

            # Save the reference to the image object in the tab data of the newly created tab.
            self.mainWindow.tabBar.setTabData(newTabBarIndex, newImage)

            # Open the selected image and remember the reference to its image object.
            newImage.openFile(filePath)


    def saveImage(self):
        """ Open a file dialog and save the current image to the chosen file path.
        """
        if self.currentImageObject:
            filePath = QFileDialog.getSaveFileName(self.mainWindow, "Save file", filter="*.png;;*.jpg;;*.tiff")[0]

            if filePath:
                # Change tab text to new filename.
                filename = os.path.split(filePath)[1]
                index = self.mainWindow.tabBar.currentIndex()
                self.mainWindow.tabBar.setTabText(index, filename)

                # Save the image to the new filename.
                self.currentImageObject.saveImage(filePath)

    def exportDiametersToCSV(self):
        if self.currentImageObject:
            if len(self.currentImageObject._diameters) == 0:
                msgbox = QMessageBox()
                msgbox.setIcon(QMessageBox.Information)
                msgbox.setWindowTitle("Python Fiber Image Analyzer")
                msgbox.setText("Diameter analysis has not been executed yet for this image.")
                msgbox.exec_()
            else:
                filePath = QFileDialog.getSaveFileName(self.mainWindow, "Export Diameter Data", filter="*.csv")[0]

                if filePath:
                    self.currentImageObject.exportDiametersToCSV(filePath)


    def exportAnglesToCSV(self):
        if self.currentImageObject:
            if len(self.currentImageObject._angles) == 0:
                msgbox = QMessageBox()
                msgbox.setIcon(QMessageBox.Information)
                msgbox.setWindowTitle("Python Fiber Image Analyzer")
                msgbox.setText("Orientation analysis has not been executed yet for this image.")
                msgbox.exec_()
            else:
                filePath = QFileDialog.getSaveFileName(self.mainWindow, "Export Orientation Angles Data", filter="*.csv")[0]

                if filePath:
                    self.currentImageObject.exportAnglesToCSV(filePath)

    def displayImage(self):
        """ Display the image model of the current tab with its zoom factor on the QLabel.
        """
        if self.currentImageObject:
            img = self.currentImageObject.getZoomedImage()

            if img is not None:
                height, width = img.shape
                qImg = QImage(img, width, height, img.strides[0], QImage.Format_Grayscale8)
                qPix = QPixmap.fromImage(qImg)
                self.mainWindow.lblImgDisplay.setPixmap(qPix)

    def zoomIn(self):
        """ Zoom in the current image by 25%.
        """
        if self.currentImageObject:
            self.currentImageObject.zoomFactor += 0.25

    def zoomOut(self):
        """ Zoom out the current image by 25%.
        """
        if self.currentImageObject:
            self.currentImageObject.zoomFactor -= 0.25

    def resetZoom(self):
        """ Reset the zoom factor to 100%
        """
        if self.currentImageObject:
            self.currentImageObject.zoomFactor = 1

    def addSharpening(self):
        """ Sharpen the current image.
        """
        if self.currentImageObject:
            self.currentImageObject.sharpen()

    def addWatershedSegmentation(self):
        """ Add watershed segmentation to the current image.
        """
        if self.currentImageObject:
            self.currentImageObject.watershedSegmentation()

    def addContour(self):
        """ Highlight the contours of the current image.
        """
        if self.currentImageObject:
            self.currentImageObject.contours()

    def update(self, *args, **kwargs):
        """ Update the current image. This method serves as slot for an observer pattern.
        """
        if self.currentImageObject:
            self.displayImage()
            self.mainWindow.lblHeight.setText(str(self.currentImageObject.getImageDimensions()[0]))
            self.mainWindow.lblWidth.setText(str(self.currentImageObject.getImageDimensions()[1]))
            self.mainWindow.lblZoomFactor.setText(str(int(self.currentImageObject.zoomFactor * 100)))
            self.mainWindow.lblScale.setText("{0:.3f}".format(self.currentImageObject.scale))

            isBinary = self.currentImageObject.isBinary()
            if isBinary:
                self.mainWindow.lblBinary.setText("Yes")
                #for action in self.mainWindow.actions
            else:
                self.mainWindow.lblBinary.setText("No")

