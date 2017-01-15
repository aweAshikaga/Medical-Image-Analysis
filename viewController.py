import view.mainWindow as mainUi
import model.model as model
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage


class ViewController(mainUi.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.model = model.Model()
        self.window = QMainWindow()
        self.setupUi(self.window)
        self.actionOpen.triggered.connect(self.openFile)
        self.actionContrast.triggered.connect(self.addContrast)
        self.window.show()
        self.model.contrast()

    def addContrast(self):
        self.model.contrast()
        self.setImage()

    def openFile(self):
        filename = QFileDialog.getOpenFileName(self.window, 'Open file', '/home')[0]
        #fn = QFileDialog.get

        if filename:
            self.model.setImage(filename)
            #self.lblImgDisplay.setPixmap(QPixmap(filename))
            #self.setImage()

    def setImage(self):
        img = self.model.getImage()
        height, width = img.shape
        qImg = QImage(img, width, height, img.strides[0], QImage.Format_Grayscale8)
        qPix = QPixmap.fromImage(qImg)
        self.lblImgDisplay.setPixmap(qPix)
