import view.Ui as Ui
import model.model as model
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication, QPainter, QColor
from PyQt5.QtCore import Qt, QEvent


class ViewController(object):
    def __init__(self):
        super().__init__()
        self.mainWindow = Ui.MainWindow()
        self.connectActions()

        self._isDefiningArea = False
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
        self.mainWindow.actionFiberOrientation.triggered.connect(self.addHughLines)
        self.mainWindow.actionTop_Hat_Transformation.triggered.connect(self.addTopHatTransformation)
        self.mainWindow.actionDilation.triggered.connect(self.addDilation)
        self.mainWindow.actionSkeletonization.triggered.connect(self.addSkeletonization)
        self.mainWindow.actionTest.triggered.connect(self.checkPorosity)
        self.mainWindow.actionDefineAreas.triggered.connect(self.grabArea)
        self.mainWindow.lblImgDisplay.setMouseTracking(True)
        self.mainWindow.lblImgDisplay.mousePressEvent = self.getMousePositionOnLabel
        self.mainWindow.lblImgDisplay.mouseMoveEvent = self.getMousePositionOnLabel
        #self.mainWindow.lblImgDisplay.paintEvent = self.paintEvent
        self.mainWindow.keyPressEvent = self.keyPressEvent

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.mainWindow.menubar.setEnabled(True)
            QGuiApplication.setOverrideCursor(Qt.ArrowCursor)
            print(self._initialAreaPoint)
            print(self._endAreaPoint)
            self.drawRectOnLabel(self._initialAreaPoint, self._endAreaPoint)
            self._isDefiningArea = False
            self.mainWindow.lblImgDisplay.repaint()
            self._initialAreaPoint = (-1, -1)
            self._endAreaPoint = (-1, -1)

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

    def grabArea(self):
        self.mainWindow.menubar.setEnabled(False)
        QGuiApplication.setOverrideCursor(Qt.CrossCursor)
        self._isDefiningArea = True

    def getMousePositionOnLabel(self, event):
        if self._isDefiningArea:
            x = event.pos().x()
            y = event.pos().y()
            print(event.button())
            if self._initialAreaPoint == (-1, -1) and event.button() == 1:
                self._initialAreaPoint = (x, y)
            else:
                self._endAreaPoint = (x, y)
                self .drawRectOnLabel(self._initialAreaPoint, self._endAreaPoint)

    def checkPorosity(self):
        """ Get the porosity of the current image.
        """
        if self.currentImageObject:
            porosity = self.currentImageObject.getPorosity() * 100
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setWindowTitle("Medical Image Analysis")
            msgbox.setText("The porosity is: {0:.2f} %".format(porosity))
            msgbox.exec_()


    def addSkeletonization(self):
        """ Add skeletonization to the current image.
        """
        if self.currentImageObject:
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

    def addHughLines(self):
        if self.currentImageObject:
            angles = self.currentImageObject.hughLines()
            np.histogram(angles)
            plt.hist(angles, bins=180, range=(0, 180))
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
                    print(contrastValue)
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

    def createImageFromArea(self, newImage):
        newTabBarIndex = self.mainWindow.tabBar.addTab("New Image")

        # Make the new tab the active tab.
        self.mainWindow.tabBar.setCurrentIndex(newTabBarIndex)

        # Create a new image object for the new tab.
        newImage = model.Image()

        # Register this viewController as an observer to the image object.
        newImage.register(self)

        # Save the reference to the image object in the tab data of the newly created tab.
        self.mainWindow.tabBar.setTabData(newTabBarIndex, newImage)

        # Open the selected image and remember the reference to its image object.
        newImage.img = newImage

    def openFile(self):
        """ Open a file dialog and open the chosen file path.
        """
        filePath = QFileDialog.getOpenFileName(self.mainWindow, 'Open file', '/home', '*.jpg; *.png; *.tiff')[0]

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
            filePath = QFileDialog.getSaveFileName(self.mainWindow, "Save file", "/home", filter="*.jpg;;*.png;;*.tiff")[0]

            if filePath:
                # Change tab text to new filename.
                filename = os.path.split(filePath)[1]
                index = self.mainWindow.tabBar.currentIndex()
                self.mainWindow.tabBar.setTabText(index, filename)

                # Save the image to the new filename.
                self.currentImageObject.saveImage(filePath)

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
            self.mainWindow.lblImageSize.setText(str(self.currentImageObject.getImageDimensions()))
            self.mainWindow.lblZoomFactor.setText(str(self.currentImageObject.zoomFactor))

