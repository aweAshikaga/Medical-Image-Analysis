import view.Ui as Ui
import model.model as model
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage


class ViewController(object):
    def __init__(self):
        super().__init__()
        self.mainWindow = Ui.MainWindow()
        self.Image = model.Image()
        self.Image.register(self)
        self.connectActions()

    def connectActions(self):
        self.mainWindow.actionOpen.triggered.connect(self.openFile)
        self.mainWindow.actionContrast.triggered.connect(self.addContrast)
        self.mainWindow.actionZoom_in.triggered.connect(self.zoomIn)
        self.mainWindow.actionZoom_out.triggered.connect(self.zoomOut)

    def addContrast(self):
        dialog = Ui.ContrastDialog()
        dialog.exec_()
        if dialog.result() == 1:
            if dialog.radioBtnCLAHE.isChecked():
                self.Image.addContrast()
            elif dialog.radioBtnManual.isChecked():
                self.Image.addContrast2(2)

    def openFile(self):
        filename = QFileDialog.getOpenFileName(self.mainWindow.window, 'Open file', '/home', '*.jpg')[0]

        if filename:
            self.Image.openFile(filename)

    def displayImage(self):
        img = self.Image.getImage()
        height, width = img.shape
        qImg = QImage(img, width, height, img.strides[0], QImage.Format_Grayscale8)
        qPix = QPixmap.fromImage(qImg)
        self.mainWindow.lblImgDisplay.setPixmap(qPix)

    def zoomIn(self):
        self.Image.zoomFactor += 0.25

    def zoomOut(self):
        self.Image.zoomFactor -= 0.25

    def update(self):
        self.displayImage()
        self.mainWindow.lblImageSize.setText(str(self.Image.getImageDimensions()))
        self.mainWindow.lblZoomFactor.setText(str(self.Image.zoomFactor))