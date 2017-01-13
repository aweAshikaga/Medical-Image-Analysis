# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.saImgDisplay = QtWidgets.QScrollArea(self.centralwidget)
        self.saImgDisplay.setWidgetResizable(True)
        self.saImgDisplay.setObjectName("saImgDisplay")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 780, 539))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lblImgDisplay = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.lblImgDisplay.setText("")
        self.lblImgDisplay.setObjectName("lblImgDisplay")
        self.verticalLayout.addWidget(self.lblImgDisplay)
        self.saImgDisplay.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.saImgDisplay)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuProcessing = QtWidgets.QMenu(self.menubar)
        self.menuProcessing.setObjectName("menuProcessing")
        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
        self.menuTest1 = QtWidgets.QMenu(self.menubar)
        self.menuTest1.setObjectName("menuTest1")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExport = QtWidgets.QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.actionUndo = QtWidgets.QAction(MainWindow)
        self.actionUndo.setObjectName("actionUndo")
        self.actionRedo = QtWidgets.QAction(MainWindow)
        self.actionRedo.setObjectName("actionRedo")
        self.actionZoom_in = QtWidgets.QAction(MainWindow)
        self.actionZoom_in.setObjectName("actionZoom_in")
        self.actionZoom_out = QtWidgets.QAction(MainWindow)
        self.actionZoom_out.setObjectName("actionZoom_out")
        self.actionDefine_scale = QtWidgets.QAction(MainWindow)
        self.actionDefine_scale.setObjectName("actionDefine_scale")
        self.actionDefine_areas = QtWidgets.QAction(MainWindow)
        self.actionDefine_areas.setObjectName("actionDefine_areas")
        self.actionFilter = QtWidgets.QAction(MainWindow)
        self.actionFilter.setObjectName("actionFilter")
        self.actionSegmentation = QtWidgets.QAction(MainWindow)
        self.actionSegmentation.setObjectName("actionSegmentation")
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.actionFiber_diameter = QtWidgets.QAction(MainWindow)
        self.actionFiber_diameter.setObjectName("actionFiber_diameter")
        self.actionFiber_orientation = QtWidgets.QAction(MainWindow)
        self.actionFiber_orientation.setObjectName("actionFiber_orientation")
        self.actionTest2 = QtWidgets.QAction(MainWindow)
        self.actionTest2.setObjectName("actionTest2")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExport)
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addAction(self.actionZoom_in)
        self.menuEdit.addAction(self.actionZoom_out)
        self.menuEdit.addAction(self.actionDefine_scale)
        self.menuEdit.addAction(self.actionDefine_areas)
        self.menuProcessing.addAction(self.actionFilter)
        self.menuProcessing.addAction(self.actionSegmentation)
        self.menuProcessing.addAction(self.actionContrast)
        self.menuAnalysis.addAction(self.actionFiber_diameter)
        self.menuAnalysis.addAction(self.actionFiber_orientation)
        self.menuTest1.addAction(self.actionTest2)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuProcessing.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuTest1.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuProcessing.setTitle(_translate("MainWindow", "Processing"))
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))
        self.menuTest1.setTitle(_translate("MainWindow", "Test1"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExport.setText(_translate("MainWindow", "Export"))
        self.actionUndo.setText(_translate("MainWindow", "Undo"))
        self.actionRedo.setText(_translate("MainWindow", "Redo"))
        self.actionZoom_in.setText(_translate("MainWindow", "Zoom in"))
        self.actionZoom_out.setText(_translate("MainWindow", "Zoom out"))
        self.actionDefine_scale.setText(_translate("MainWindow", "Define scale"))
        self.actionDefine_areas.setText(_translate("MainWindow", "Define areas"))
        self.actionFilter.setText(_translate("MainWindow", "Filter"))
        self.actionSegmentation.setText(_translate("MainWindow", "Segmentation"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.actionFiber_diameter.setText(_translate("MainWindow", "Fiber diameter"))
        self.actionFiber_orientation.setText(_translate("MainWindow", "Fiber orientation"))
        self.actionTest2.setText(_translate("MainWindow", "test2"))

