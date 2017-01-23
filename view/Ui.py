from view.mainWindow_structure import Ui_MainWindow
from view.contrastDialog_structure import Ui_ContrastDialog
from PyQt5.QtWidgets import QMainWindow, QDialog


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

class ContrastDialog(QDialog, Ui_ContrastDialog):
    def __init__(self):
        QDialog.__init__(self)
        Ui_ContrastDialog.__init__(self)
        self.setupUi(self)