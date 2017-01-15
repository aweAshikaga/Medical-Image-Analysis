import sys

from PyQt5.QtWidgets import QApplication

import viewController as vc
#test
app = QApplication(sys.argv)
viewController = vc.ViewController()

sys.exit(app.exec())