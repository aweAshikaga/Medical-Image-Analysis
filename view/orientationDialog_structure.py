# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'orientationDialog.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_orientationDialog(object):
    def setupUi(self, orientationDialog):
        orientationDialog.setObjectName("orientationDialog")
        orientationDialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(orientationDialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.lblThresholdText = QtWidgets.QLabel(orientationDialog)
        self.lblThresholdText.setGeometry(QtCore.QRect(40, 50, 71, 16))
        self.lblThresholdText.setObjectName("lblThresholdText")
        self.doubleSpinBoxThreshold = QtWidgets.QDoubleSpinBox(orientationDialog)
        self.doubleSpinBoxThreshold.setGeometry(QtCore.QRect(180, 50, 62, 22))
        self.doubleSpinBoxThreshold.setMaximum(1.0)
        self.doubleSpinBoxThreshold.setSingleStep(0.05)
        self.doubleSpinBoxThreshold.setProperty("value", 0.5)
        self.doubleSpinBoxThreshold.setObjectName("doubleSpinBoxThreshold")
        self.lblLineCount = QtWidgets.QLabel(orientationDialog)
        self.lblLineCount.setGeometry(QtCore.QRect(40, 110, 151, 41))
        self.lblLineCount.setWordWrap(True)
        self.lblLineCount.setObjectName("lblLineCount")
        self.spinBoxMaxLines = QtWidgets.QSpinBox(orientationDialog)
        self.spinBoxMaxLines.setGeometry(QtCore.QRect(180, 110, 81, 22))
        self.spinBoxMaxLines.setMaximum(1000)
        self.spinBoxMaxLines.setObjectName("spinBoxMaxLines")

        self.retranslateUi(orientationDialog)
        self.buttonBox.accepted.connect(orientationDialog.accept)
        self.buttonBox.rejected.connect(orientationDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(orientationDialog)

    def retranslateUi(self, orientationDialog):
        _translate = QtCore.QCoreApplication.translate
        orientationDialog.setWindowTitle(_translate("orientationDialog", "Orientation Analysis"))
        self.lblThresholdText.setText(_translate("orientationDialog", "Threshold:"))
        self.lblLineCount.setText(_translate("orientationDialog", "<html><head/><body><p>Maximum number of lines:</p><p>(0 for infinity)</p></body></html>"))

