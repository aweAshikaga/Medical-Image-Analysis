from view.mainWindow_structure import Ui_MainWindow
from view.contrastDialog_structure import Ui_ContrastDialog
from PyQt5.QtWidgets import QMainWindow, QDialog


class MainWindow(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.window = QMainWindow()
        self.setupUi(self.window)
        self.window.show()


class ContrastDialog(QDialog, Ui_ContrastDialog):
    def __init__(self):
        QDialog.__init__(self)
        Ui_ContrastDialog.__init__(self)
        self.setupUi(self)