import view.Ui as Ui
import model.model as model
import os
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage


class ViewController(object):
    def __init__(self):
        super().__init__()
        self.mainWindow = Ui.MainWindow()
        self.connectActions()

    @property
    def currentImageObject(self):
        return self.mainWindow.tabBar.tabData(self.mainWindow.tabBar.currentIndex())

    def connectActions(self):
        self.mainWindow.actionOpen.triggered.connect(self.openFile)
        self.mainWindow.actionContrast.triggered.connect(self.addContrast)
        self.mainWindow.actionZoom_in.triggered.connect(self.zoomIn)
        self.mainWindow.actionZoom_out.triggered.connect(self.zoomOut)
        self.mainWindow.actionRedo.triggered.connect(self.redo)
        self.mainWindow.actionUndo.triggered.connect(self.undo)
        self.mainWindow.tabBar.currentChanged.connect(self.update)

    def redo(self):
        if self.currentImageObject:
            self.currentImageObject.redo()

    def undo(self):
        if self.currentImageObject:
            self.currentImageObject.undo()

    def addContrast(self):
        if self.currentImageObject:
            dialog = Ui.ContrastDialog()
            dialog.exec_()
            if dialog.result() == 1:
                if dialog.radioBtnCLAHE.isChecked():
                    self.currentImageObject.addContrastCLAHE()
                elif dialog.radioBtnManual.isChecked():
                    self.currentImageObject.addContrastCustom(2)

    def openFile(self):
        filePath = QFileDialog.getOpenFileName(self.mainWindow.window, 'Open file', '/home', '*.jpg; *.png')[0]

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

    def displayImage(self):
        if self.currentImageObject:
            img = self.currentImageObject.getZoomedImage()

            if img is not None:
                height, width = img.shape
                qImg = QImage(img, width, height, img.strides[0], QImage.Format_Grayscale8)
                qPix = QPixmap.fromImage(qImg)
                self.mainWindow.lblImgDisplay.setPixmap(qPix)

    def zoomIn(self):
        if self.currentImageObject:
            self.currentImageObject.zoomFactor += 0.25

    def zoomOut(self):
        if self.currentImageObject:
            self.currentImageObject.zoomFactor -= 0.25

    def update(self, *args, **kwargs):
        if self.currentImageObject:
            self.displayImage()
            self.mainWindow.lblImageSize.setText(str(self.currentImageObject.getImageDimensions()))
            self.mainWindow.lblZoomFactor.setText(str(self.currentImageObject.zoomFactor))
