# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'contrastDialog.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ContrastDialog(object):
    def setupUi(self, ContrastDialog):
        ContrastDialog.setObjectName("ContrastDialog")
        ContrastDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        ContrastDialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(ContrastDialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.radioBtnCLAHE = QtWidgets.QRadioButton(ContrastDialog)
        self.radioBtnCLAHE.setGeometry(QtCore.QRect(30, 50, 281, 17))
        self.radioBtnCLAHE.setChecked(True)
        self.radioBtnCLAHE.setObjectName("radioBtnCLAHE")
        self.radioBtnManual = QtWidgets.QRadioButton(ContrastDialog)
        self.radioBtnManual.setGeometry(QtCore.QRect(30, 150, 321, 17))
        self.radioBtnManual.setObjectName("radioBtnManual")
        self.label = QtWidgets.QLabel(ContrastDialog)
        self.label.setGeometry(QtCore.QRect(80, 80, 47, 13))
        self.label.setObjectName("label")
        self.spinBox = QtWidgets.QSpinBox(ContrastDialog)
        self.spinBox.setGeometry(QtCore.QRect(130, 80, 42, 22))
        self.spinBox.setProperty("value", 1)
        self.spinBox.setObjectName("spinBox")
        self.label_2 = QtWidgets.QLabel(ContrastDialog)
        self.label_2.setGeometry(QtCore.QRect(80, 180, 47, 13))
        self.label_2.setObjectName("label_2")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(ContrastDialog)
        self.doubleSpinBox.setGeometry(QtCore.QRect(130, 180, 62, 22))
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMaximum(10.0)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.setObjectName("doubleSpinBox")

        self.retranslateUi(ContrastDialog)
        self.buttonBox.accepted.connect(ContrastDialog.accept)
        self.buttonBox.rejected.connect(ContrastDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ContrastDialog)

    def retranslateUi(self, ContrastDialog):
        _translate = QtCore.QCoreApplication.translate
        ContrastDialog.setWindowTitle(_translate("ContrastDialog", "Dialog"))
        self.radioBtnCLAHE.setText(_translate("ContrastDialog", "Contrast limited adaptive histogram equalisation"))
        self.radioBtnManual.setText(_translate("ContrastDialog", "Manual contrast value"))
        self.label.setText(_translate("ContrastDialog", "clip limit:"))
        self.label_2.setText(_translate("ContrastDialog", "value:"))

