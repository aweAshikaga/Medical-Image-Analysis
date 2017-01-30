# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'segmentationDialog.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SegmentationDialog(object):
    def setupUi(self, SegmentationDialog):
        SegmentationDialog.setObjectName("SegmentationDialog")
        SegmentationDialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(SegmentationDialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 240, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.radioBtnManual = QtWidgets.QRadioButton(SegmentationDialog)
        self.radioBtnManual.setGeometry(QtCore.QRect(40, 40, 161, 17))
        self.radioBtnManual.setChecked(True)
        self.radioBtnManual.setObjectName("radioBtnManual")
        self.radioBtnAdaptive = QtWidgets.QRadioButton(SegmentationDialog)
        self.radioBtnAdaptive.setGeometry(QtCore.QRect(40, 120, 171, 17))
        self.radioBtnAdaptive.setObjectName("radioBtnAdaptive")
        self.radioBtnOtsu = QtWidgets.QRadioButton(SegmentationDialog)
        self.radioBtnOtsu.setGeometry(QtCore.QRect(40, 150, 181, 17))
        self.radioBtnOtsu.setObjectName("radioBtnOtsu")
        self.hSliderManual = QtWidgets.QSlider(SegmentationDialog)
        self.hSliderManual.setEnabled(True)
        self.hSliderManual.setGeometry(QtCore.QRect(70, 70, 301, 22))
        self.hSliderManual.setMaximum(255)
        self.hSliderManual.setProperty("value", 127)
        self.hSliderManual.setOrientation(QtCore.Qt.Horizontal)
        self.hSliderManual.setInvertedAppearance(False)
        self.hSliderManual.setInvertedControls(False)
        self.hSliderManual.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.hSliderManual.setObjectName("hSliderManual")
        self.lblManualValue = QtWidgets.QLabel(SegmentationDialog)
        self.lblManualValue.setGeometry(QtCore.QRect(210, 100, 21, 16))
        self.lblManualValue.setObjectName("lblManualValue")

        self.retranslateUi(SegmentationDialog)
        self.buttonBox.accepted.connect(SegmentationDialog.accept)
        self.buttonBox.rejected.connect(SegmentationDialog.reject)
        self.hSliderManual.valueChanged['int'].connect(self.lblManualValue.setNum)
        self.radioBtnManual.toggled['bool'].connect(self.hSliderManual.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(SegmentationDialog)

    def retranslateUi(self, SegmentationDialog):
        _translate = QtCore.QCoreApplication.translate
        SegmentationDialog.setWindowTitle(_translate("SegmentationDialog", "Segmentation"))
        self.radioBtnManual.setText(_translate("SegmentationDialog", "Manual threshold"))
        self.radioBtnAdaptive.setText(_translate("SegmentationDialog", "Adaptive threshold"))
        self.radioBtnOtsu.setText(_translate("SegmentationDialog", "Otsu threshold"))
        self.lblManualValue.setText(_translate("SegmentationDialog", "127"))

