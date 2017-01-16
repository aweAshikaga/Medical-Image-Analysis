import sys

from PyQt5.QtWidgets import QApplication

import viewController as vc

app = QApplication(sys.argv)
viewController = vc.ViewController()

sys.exit(app.exec())