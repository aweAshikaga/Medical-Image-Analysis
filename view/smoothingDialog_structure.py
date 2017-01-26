# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'smoothingDialog.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SmoothingDialog(object):
    def setupUi(self, SmoothingDialog):
        SmoothingDialog.setObjectName("SmoothingDialog")
        SmoothingDialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(SmoothingDialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.radioBtnAverage = QtWidgets.QRadioButton(SmoothingDialog)
        self.radioBtnAverage.setGeometry(QtCore.QRect(30, 70, 82, 17))
        self.radioBtnAverage.setChecked(True)
        self.radioBtnAverage.setObjectName("radioBtnAverage")
        self.radioBtnGaussian = QtWidgets.QRadioButton(SmoothingDialog)
        self.radioBtnGaussian.setGeometry(QtCore.QRect(30, 100, 82, 17))
        self.radioBtnGaussian.setObjectName("radioBtnGaussian")
        self.radioBtnMedian = QtWidgets.QRadioButton(SmoothingDialog)
        self.radioBtnMedian.setGeometry(QtCore.QRect(30, 130, 82, 17))
        self.radioBtnMedian.setObjectName("radioBtnMedian")
        self.spinBoxKernelSize = QtWidgets.QSpinBox(SmoothingDialog)
        self.spinBoxKernelSize.setGeometry(QtCore.QRect(100, 170, 41, 22))
        self.spinBoxKernelSize.setSuffix("")
        self.spinBoxKernelSize.setMinimum(1)
        self.spinBoxKernelSize.setSingleStep(1)
        self.spinBoxKernelSize.setProperty("value", 5)
        self.spinBoxKernelSize.setObjectName("spinBoxKernelSize")
        self.lblKernelSize = QtWidgets.QLabel(SmoothingDialog)
        self.lblKernelSize.setGeometry(QtCore.QRect(30, 170, 61, 16))
        self.lblKernelSize.setObjectName("lblKernelSize")
        self.lblDescription = QtWidgets.QLabel(SmoothingDialog)
        self.lblDescription.setGeometry(QtCore.QRect(30, 20, 341, 31))
        self.lblDescription.setTextFormat(QtCore.Qt.AutoText)
        self.lblDescription.setScaledContents(False)
        self.lblDescription.setWordWrap(True)
        self.lblDescription.setObjectName("lblDescription")

        self.retranslateUi(SmoothingDialog)
        QtCore.QMetaObject.connectSlotsByName(SmoothingDialog)

    def retranslateUi(self, SmoothingDialog):
        _translate = QtCore.QCoreApplication.translate
        SmoothingDialog.setWindowTitle(_translate("SmoothingDialog", "Smoothing Filter"))
        self.radioBtnAverage.setText(_translate("SmoothingDialog", "Average"))
        self.radioBtnGaussian.setText(_translate("SmoothingDialog", "Gaussian"))
        self.radioBtnMedian.setText(_translate("SmoothingDialog", "Median"))
        self.lblKernelSize.setText(_translate("SmoothingDialog", "Kernel size:"))
        self.lblDescription.setText(_translate("SmoothingDialog", "Please select a smoothing method and a kernel size (kernel size must be a positive, odd number)."))

