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
        MainWindow.resize(799, 596)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabBar = QTabBar(self.centralwidget)
        self.tabBar.setObjectName("tabBar")
        self.verticalLayout_2.addWidget(self.tabBar)
        self.saImgDisplay = QtWidgets.QScrollArea(self.centralwidget)
        self.saImgDisplay.setWidgetResizable(True)
        self.saImgDisplay.setObjectName("saImgDisplay")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 641, 493))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lblImgDisplay = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.lblImgDisplay.setText("")
        self.lblImgDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.lblImgDisplay.setObjectName("lblImgDisplay")
        self.horizontalLayout.addWidget(self.lblImgDisplay)
        self.saImgDisplay.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.saImgDisplay)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.groupBoxInfo = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxInfo.setMinimumSize(QtCore.QSize(130, 0))
        self.groupBoxInfo.setMaximumSize(QtCore.QSize(150, 16777215))
        self.groupBoxInfo.setObjectName("groupBoxInfo")
        self.formLayout = QtWidgets.QFormLayout(self.groupBoxInfo)
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout.setObjectName("formLayout")
        self.lblWidthText = QtWidgets.QLabel(self.groupBoxInfo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblWidthText.sizePolicy().hasHeightForWidth())
        self.lblWidthText.setSizePolicy(sizePolicy)
        self.lblWidthText.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lblWidthText.setObjectName("lblWidthText")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.lblWidthText)
        self.lblWidth = QtWidgets.QLabel(self.groupBoxInfo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblWidth.sizePolicy().hasHeightForWidth())
        self.lblWidth.setSizePolicy(sizePolicy)
        self.lblWidth.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lblWidth.setObjectName("lblWidth")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lblWidth)
        self.lblZoomFactorText = QtWidgets.QLabel(self.groupBoxInfo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblZoomFactorText.sizePolicy().hasHeightForWidth())
        self.lblZoomFactorText.setSizePolicy(sizePolicy)
        self.lblZoomFactorText.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lblZoomFactorText.setObjectName("lblZoomFactorText")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.lblZoomFactorText)
        self.lblZoomFactor = QtWidgets.QLabel(self.groupBoxInfo)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lblZoomFactor.sizePolicy().hasHeightForWidth())
        self.lblZoomFactor.setSizePolicy(sizePolicy)
        self.lblZoomFactor.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.lblZoomFactor.setObjectName("lblZoomFactor")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lblZoomFactor)
        self.lblBinaryText = QtWidgets.QLabel(self.groupBoxInfo)
        self.lblBinaryText.setObjectName("lblBinaryText")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.lblBinaryText)
        self.lblBinary = QtWidgets.QLabel(self.groupBoxInfo)
        self.lblBinary.setObjectName("lblBinary")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lblBinary)
        self.lblScaleText = QtWidgets.QLabel(self.groupBoxInfo)
        self.lblScaleText.setObjectName("lblScaleText")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.lblScaleText)
        self.lblScale = QtWidgets.QLabel(self.groupBoxInfo)
        self.lblScale.setObjectName("lblScale")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lblScale)
        self.lblHeightText = QtWidgets.QLabel(self.groupBoxInfo)
        self.lblHeightText.setObjectName("lblHeightText")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lblHeightText)
        self.lblHeight = QtWidgets.QLabel(self.groupBoxInfo)
        self.lblHeight.setObjectName("lblHeight")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lblHeight)
        self.horizontalLayout_2.addWidget(self.groupBoxInfo)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 799, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuProcessing = QtWidgets.QMenu(self.menubar)
        self.menuProcessing.setObjectName("menuProcessing")
        self.menuDerivative_Filter = QtWidgets.QMenu(self.menuProcessing)
        self.menuDerivative_Filter.setObjectName("menuDerivative_Filter")
        self.menuMorphological_Transformation = QtWidgets.QMenu(self.menuProcessing)
        self.menuMorphological_Transformation.setObjectName("menuMorphological_Transformation")
        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setObjectName("menuAnalysis")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/view/res/icons/document-open.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExport_diameters = QtWidgets.QAction(MainWindow)
        self.actionExport_diameters.setObjectName("actionExport_diameters")
        self.actionUndo = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/view/res/icons/edit-undo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionUndo.setIcon(icon1)
        self.actionUndo.setObjectName("actionUndo")
        self.actionRedo = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/view/res/icons/edit-redo.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRedo.setIcon(icon2)
        self.actionRedo.setObjectName("actionRedo")
        self.actionZoomIn = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/view/res/icons/zoom-in.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoomIn.setIcon(icon3)
        self.actionZoomIn.setObjectName("actionZoomIn")
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/view/res/icons/zoom-out.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoomOut.setIcon(icon4)
        self.actionZoomOut.setObjectName("actionZoomOut")
        self.actionDefineScale = QtWidgets.QAction(MainWindow)
        self.actionDefineScale.setObjectName("actionDefineScale")
        self.actionDefineAreas = QtWidgets.QAction(MainWindow)
        self.actionDefineAreas.setObjectName("actionDefineAreas")
        self.actionSegmentation = QtWidgets.QAction(MainWindow)
        self.actionSegmentation.setObjectName("actionSegmentation")
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.actionFiberDiameter = QtWidgets.QAction(MainWindow)
        self.actionFiberDiameter.setObjectName("actionFiberDiameter")
        self.actionFiberOrientation = QtWidgets.QAction(MainWindow)
        self.actionFiberOrientation.setObjectName("actionFiberOrientation")
        self.actionDerivativeSobel = QtWidgets.QAction(MainWindow)
        self.actionDerivativeSobel.setObjectName("actionDerivativeSobel")
        self.actionDerivativeLaPlace = QtWidgets.QAction(MainWindow)
        self.actionDerivativeLaPlace.setObjectName("actionDerivativeLaPlace")
        self.actionSmoothingFilter = QtWidgets.QAction(MainWindow)
        self.actionSmoothingFilter.setObjectName("actionSmoothingFilter")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionWatershed_Transformation = QtWidgets.QAction(MainWindow)
        self.actionWatershed_Transformation.setObjectName("actionWatershed_Transformation")
        self.actionEdgeDetection = QtWidgets.QAction(MainWindow)
        self.actionEdgeDetection.setObjectName("actionEdgeDetection")
        self.actionTop_Hat_Transformation = QtWidgets.QAction(MainWindow)
        self.actionTop_Hat_Transformation.setObjectName("actionTop_Hat_Transformation")
        self.actionDilation = QtWidgets.QAction(MainWindow)
        self.actionDilation.setObjectName("actionDilation")
        self.actionSkeletonization = QtWidgets.QAction(MainWindow)
        self.actionSkeletonization.setObjectName("actionSkeletonization")
        self.actionPorosity = QtWidgets.QAction(MainWindow)
        self.actionPorosity.setObjectName("actionPorosity")
        self.actionExport_orientation_angles = QtWidgets.QAction(MainWindow)
        self.actionExport_orientation_angles.setObjectName("actionExport_orientation_angles")
        self.actionReset_zoom = QtWidgets.QAction(MainWindow)
        self.actionReset_zoom.setObjectName("actionReset_zoom")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExport_diameters)
        self.menuFile.addAction(self.actionExport_orientation_angles)
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addAction(self.actionZoomIn)
        self.menuEdit.addAction(self.actionZoomOut)
        self.menuEdit.addAction(self.actionReset_zoom)
        self.menuEdit.addAction(self.actionDefineScale)
        self.menuEdit.addAction(self.actionDefineAreas)
        self.menuDerivative_Filter.addAction(self.actionDerivativeSobel)
        self.menuDerivative_Filter.addAction(self.actionDerivativeLaPlace)
        self.menuMorphological_Transformation.addAction(self.actionTop_Hat_Transformation)
        self.menuMorphological_Transformation.addAction(self.actionDilation)
        self.menuMorphological_Transformation.addAction(self.actionSkeletonization)
        self.menuProcessing.addAction(self.actionSmoothingFilter)
        self.menuProcessing.addAction(self.menuDerivative_Filter.menuAction())
        self.menuProcessing.addAction(self.actionSegmentation)
        self.menuProcessing.addAction(self.actionContrast)
        self.menuProcessing.addAction(self.actionSharpen)
        self.menuProcessing.addAction(self.actionWatershed_Transformation)
        self.menuProcessing.addAction(self.actionEdgeDetection)
        self.menuProcessing.addAction(self.menuMorphological_Transformation.menuAction())
        self.menuAnalysis.addAction(self.actionFiberDiameter)
        self.menuAnalysis.addAction(self.actionFiberOrientation)
        self.menuAnalysis.addAction(self.actionPorosity)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuProcessing.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionUndo)
        self.toolBar.addAction(self.actionRedo)
        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBoxInfo.setTitle(_translate("MainWindow", "Information"))
        self.lblWidthText.setText(_translate("MainWindow", "Width [Px]:"))
        self.lblWidth.setText(_translate("MainWindow", "0"))
        self.lblZoomFactorText.setText(_translate("MainWindow", "Zoom [%]:"))
        self.lblZoomFactor.setText(_translate("MainWindow", "-"))
        self.lblBinaryText.setText(_translate("MainWindow", "Binary:"))
        self.lblBinary.setText(_translate("MainWindow", "-"))
        self.lblScaleText.setText(_translate("MainWindow", "Scale [um/Px]:"))
        self.lblScale.setText(_translate("MainWindow", "-"))
        self.lblHeightText.setText(_translate("MainWindow", "Height [Px]:"))
        self.lblHeight.setText(_translate("MainWindow", "0"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuProcessing.setTitle(_translate("MainWindow", "Processing"))
        self.menuDerivative_Filter.setTitle(_translate("MainWindow", "Derivative Filter"))
        self.menuMorphological_Transformation.setTitle(_translate("MainWindow", "Morphological Transformation"))
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExport_diameters.setText(_translate("MainWindow", "Export diameters"))
        self.actionUndo.setText(_translate("MainWindow", "Undo"))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z"))
        self.actionRedo.setText(_translate("MainWindow", "Redo"))
        self.actionRedo.setShortcut(_translate("MainWindow", "Ctrl+Y"))
        self.actionZoomIn.setText(_translate("MainWindow", "Zoom in"))
        self.actionZoomOut.setText(_translate("MainWindow", "Zoom out"))
        self.actionDefineScale.setText(_translate("MainWindow", "Define scale"))
        self.actionDefineAreas.setText(_translate("MainWindow", "Define areas"))
        self.actionSegmentation.setText(_translate("MainWindow", "Segmentation"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.actionFiberDiameter.setText(_translate("MainWindow", "Fiber diameter"))
        self.actionFiberOrientation.setText(_translate("MainWindow", "Fiber orientation"))
        self.actionDerivativeSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionDerivativeLaPlace.setText(_translate("MainWindow", "LaPlace"))
        self.actionSmoothingFilter.setText(_translate("MainWindow", "Smoothing Filter"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionWatershed_Transformation.setText(_translate("MainWindow", "Watershed Transformation"))
        self.actionEdgeDetection.setText(_translate("MainWindow", "Edge detection"))
        self.actionTop_Hat_Transformation.setText(_translate("MainWindow", "Top-Hat Transformation"))
        self.actionDilation.setText(_translate("MainWindow", "Dilation"))
        self.actionSkeletonization.setText(_translate("MainWindow", "Skeletonization"))
        self.actionPorosity.setText(_translate("MainWindow", "Porosity"))
        self.actionExport_orientation_angles.setText(_translate("MainWindow", "Export orientation angles"))
        self.actionReset_zoom.setText(_translate("MainWindow", "Reset zoom"))

from PyQt5.QtWidgets import QTabBar
import resources_rc
