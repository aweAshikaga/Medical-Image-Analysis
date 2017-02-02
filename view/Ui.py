from view.mainWindow_structure import Ui_MainWindow
from view.contrastDialog_structure import Ui_ContrastDialog
from view.smoothingDialog_structure import Ui_SmoothingDialog
from view.segmentationDialog_structure import Ui_SegmentationDialog
from PyQt5.QtWidgets import QMainWindow, QDialog, QMessageBox


class MainWindow(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.window = QMainWindow()
        self.setupUi(self.window)
        self.setupTabBar()
        self.window.show()

    def setupTabBar(self):
        """ The QT-designer does not provide the QTabBar-widget for some reason.
            That is why a QT-widget was promoted to a QTabBar but could not be set up in the designer.
            This function sets up all necessary properties for the QTabBar.
        """
        self.tabBar.setMovable(True)
        self.tabBar.setTabsClosable(True)
        self.tabBar.tabCloseRequested.connect(self.closeTab)

    def closeTab(self, index):
        self.tabBar.removeTab(index)


class ContrastDialog(QDialog, Ui_ContrastDialog):
    def __init__(self):
        QDialog.__init__(self)
        Ui_ContrastDialog.__init__(self)
        self.setupUi(self)


class SmoothingDialog(QDialog, Ui_SmoothingDialog):
    def __init__(self):
        QDialog.__init__(self)
        Ui_SmoothingDialog.__init__(self)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.acceptButtonClicked)
        self.buttonBox.rejected.connect(self.reject)

    def acceptButtonClicked(self):
        # Check if the kernel size has a valid value
        if (self.radioBtnMedian.isChecked() or self.radioBtnGaussian.isChecked()) and self.spinBoxKernelSize.value() % 2 == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Medical Image Analysis")
            msgBox.setText("The kernel size for the median filter must be a positive, odd whole number")
            msgBox.exec_()
        elif self.spinBoxKernelSize.value() < 1:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Medical Image Analysis")
            msgBox.setText("The kernel size for the median filter must be a positive whole number")
            msgBox.exec_()
        else:
            self.accept()

class SegmentationDialog(QDialog, Ui_SegmentationDialog):
    def __init__(self):
        QDialog.__init__(self)
        Ui_SegmentationDialog.__init__(self)
        self.setupUi(self)

