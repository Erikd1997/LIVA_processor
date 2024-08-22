# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:57:52 2024

Script to use a qt GUI in combination with the near-surface velocity
reconstruction

@author: ErikD
"""
# Python-core modules
import sys
from pathlib import Path
import logging
import random
import re
import datetime
import os
import configparser
import inspect
cwdir = os.path.abspath('')
os.chdir(cwdir)


# Well-known python modules
import numpy as np
import pandas as pd

# Third party module, PyQt5 for application
from PyQt5 import QtCore, QtWidgets, QtGui

# Third party module for geometry manipulation and visualisation
import vtk
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Third party module to be wrapped around vtk tools
import pyvista as pv
from pyvistaqt import QtInteractor

# Own modules
import pivDataAssimilationSurfaceParticlesFraction_SkinFriction as pivDA
from applicationHelperClasses import (QtTabWidgetCollapsible,
                                      MouseInteractorStyle,
                                      QtTextEditLogger,
                                      QtWaitingSpinner,
                                      QSettingsWindow)
from applicationSetupCheck import ProgressBarSetupDialog
from applicationThreadedWorkers import LoadFluidDataWorker, MainSolverWorker

# =============================================================================
# ### Function definitions
# =============================================================================

def ifempty(val1, val2):
    return val1 if val1 != '' else val2

def if_config_empty(config, key, default):
    try:
        out = config[key]
    except KeyError:
        out = default
    finally:
        return out
    
def if_config_empty2(config, key1, key2, default):
    try:
        out = config[key1][key2]
    except KeyError:
        out = default
    finally:
        return out


def get_attribute_name_from_class_as_dir(class_object, attribute_object):
    for name in class_object.__dir__():
        if getattr(class_object, name) is attribute_object:
            return name
    return None
    
# =============================================================================
# ### Class definitions
# =============================================================================

class MainWindow(QtWidgets.QMainWindow):
    # Defining some widget constants
    _baseProgressText = 'It is very quiet...'
    _NUMBER_OF_CHARACTERS_IN_STRING = 37
    _ini_filename = 'LIVA_processor.ini'
 
    def __init__(self, parent = None):
        # Instantiate parent class
        super(MainWindow, self).__init__(parent)
        
        # Resize the main window
        w = 1600
        h = 900
        self.resize(w, h)
        
        # Initialise settings from ini-file
        self.InitialiseSettings()
        
        
        # Initialise some vtk objects
        self.InitialiseVTKObjects()
        
        self.screenShotDir = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Thesis figures\LIVA-processor screenshots"
        
        
        ### Create widgets on main tab
        self.CreateViewerTabsWidget()
        self.CreateControllerTabsWidget()
        self.CreateLogViewer()
        

        # Create the overlay widget
        # self.overlay_widget = OverlayWidget(self.viewerTabsWidget, self.controllerTabsWidget)
        
        ### Add widgets to main group
        self.mainWidget = QtWidgets.QWidget()
        empty_widget = QtWidgets.QWidget()
        empty_widget.setMaximumWidth(self.controllerTabsWidget.tabBar().width() + self.controllerTabsWidget.expand_button.width())
        
        # self.mainLayout = QtWidgets.QGridLayout(self.mainWidget)
        self.mainLayout = QtWidgets.QHBoxLayout(self.mainWidget)
        # self.mainLayout.addWidget(self.viewerPlusController)
        self.mainLayout.addWidget(self.viewerTabsWidget)#, 0, 0, -1, 5)
        self.mainLayout.addWidget(self.controllerTabsWidget)#, 0, 3, -1, 2)
        # self.mainLayout.addWidget(self.overlay_widget)
        self.mainLayout.addWidget(self.loggingViewWidget)#, 0, 5, -1, 2)
        
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)
        
        ### Create menu bar
        self.CreateMenuBar()
        
        self.setWindowTitle("Near-surface Fluid Dynamics Processor")
        
        # Create a threaded pool manager to offload tasks to other threads
        self.threadManager = QtCore.QThreadPool()

        logging.info('Application loaded')        
        
        
        font = self.font()
        font.setPointSize(8)
        self.setFont(font)
        
    def CreateBaseIniFile(self):
        # Add data for each section
        self.config['Main'] = {'FluidDataDir': 'C:/',
                               'OutputMeshDir': 'C:/',
                               'TracerDataDir': 'C:/',
                               'STLDir': 'C:/',
                               'GroundPlaneFitDir': 'C:/',
                               'ScreenshotDir': 'C:/',
                               }
        
        self.config['QuickSettings_Tracer'] = {'MethodNumber': '0'}
        
        self.config['QuickSettings_Bins'] = {'MethodNumber': '4'}
        
        self.config['Method_0'] = {'sphere_radius': 8.0}
        
        self.config['Method_1'] = {'sphere_radius': 8.0}
        
        self.config['Method_2'] = {'sphere_radius': 8.0}
        
        self.config['Method_3'] = {'sphere_radius': 8.0}
        
        self.config['Method_4'] = {'sphere_radius': 8.0}
        
        self.config['Method_GT'] = {'sphere_radius': 8.0}
        
        # Use the config to create a base
        with open(self._ini_filename, 'w') as configfile:
            self.config.write(configfile)
        
    def InitialiseSettings(self):
        # Initialising the settings goes in two parts
        # Set up the config object
        self.config = configparser.ConfigParser()
        
        # Check if file exists
        filesInDir = os.listdir('.')
        
        #2. Check if current settings coincide with one of the files
        configFilePresent = self._ini_filename in filesInDir
        if configFilePresent:
            # Read the config file
            self.config.read(self._ini_filename)
        else:
            # Create the config file
            self.CreateBaseIniFile()

        # Get user directory
        user_dir = Path.home()

        #### Load the data from the config file
        # Directories
        self.fluidDataDir = if_config_empty2(self.config, 'Main', 'FluidDataDir', r'C:/')
        self.tracerDataDir = if_config_empty2(self.config, 'Main', 'TracerDataDir', r'C:/')
        self.outputMeshDir = if_config_empty2(self.config, 'Main', 'OutputMeshDir', r'C:/')
        self.STLDir = if_config_empty2(self.config, 'Main', 'STLDir', r'C:/')
        self.groundPlaneFitFolder = if_config_empty2(self.config, 'Main', 'GroundPlaneFitDir', '')
        self.screenshotDir = if_config_empty2(self.config, 'Main', 'ScreenshotDir', (user_dir / 'Pictures' / 'Screenshots').as_posix())
            
        
        # Method settings
        
        # Quick-setting of data
        
        #### Create the settings widget
        self.settings_widget = QSettingsWindow(parent=self,
                                               relative_size=(0.6, 0.6),
                                               name="Settings")
        
        self.CreateDirectorySettings()
        self.CreateSizeSettings()
        # icon = self.style().standardIcon('QtWidgets.QStyle.SP_DirIcon')
        iconFolder = QtGui.QIcon('icons/folder.svg')
        iconRuler = QtGui.QIcon('icons/ruler.svg')
        self.settings_widget.addEntry('Directories', self.directory_settings,
                                      entryIcon = iconFolder)
        self.settings_widget.addEntry('Defaulty sizes', self.size_settings,
                                      entryIcon = iconRuler)
        
    def CreateSizeSettings(self):
        self.size_settings = QtWidgets.QWidget()
        self.size_settings_layout = QtWidgets.QVBoxLayout()
        
        self.size_settings.setLayout(self.size_settings_layout)
        
    def CreateDirectorySettings(self):
        # Create widget & layout
        self.directory_settings = QtWidgets.QWidget()
        self.directory_settings_layout = QtWidgets.QVBoxLayout()
        
        # Create widgets for all directories
        directories_to_add = {'Fluid data directory': self.fluidDataDir,
                              'Tracer data directory': self.tracerDataDir,
                              'Output-mesh directory': self.outputMeshDir,
                              'STL directory': self.STLDir,
                              'Ground plane save directory': self.groundPlaneFitFolder,
                              'Screenshot save directory': self.screenshotDir}
        
        # Use one groupbox to store all label, line edit and choose directory button
        self.directory_settings_groupboxes = []
        main_groupbox = QtWidgets.QGroupBox()
        main_groupbox_layout = QtWidgets.QGridLayout()
        for idx, dir_entry in enumerate(directories_to_add.keys()):
            # Create a label
            label = QtWidgets.QLabel(dir_entry)
            
            # Create a line edit
            line_edit = QtWidgets.QLineEdit()
            line_edit.setReadOnly(True)
            line_edit.setText(directories_to_add[dir_entry])
            
            # Create an open file directory box
            choose_dir_button = QtWidgets.QPushButton("Choose directory")
            attribute_name = get_attribute_name_from_class_as_dir(self, directories_to_add[dir_entry])
            choose_dir_button.pressed.connect(lambda: self.openFileDialogToSetAttribute(directories_to_add[dir_entry],
                                                                                        dir_entry,
                                                                                        attribute_name,
                                                                                        line_edit
                                                                                        )
                                              )
            
            # Add all to a single groupbox
            groupbox = QtWidgets.QGroupBox()
            groupbox_layout = QtWidgets.QHBoxLayout()
            groupbox_layout.addWidget(label)
            groupbox_layout.addWidget(line_edit)
            groupbox_layout.addWidget(choose_dir_button)
            groupbox.setLayout(groupbox_layout)
            
            # # Add to the layout
            main_groupbox_layout.addWidget(label, idx, 0, 1, 1)
            main_groupbox_layout.addWidget(line_edit, idx, 1, 1, 1)
            main_groupbox_layout.addWidget(choose_dir_button, idx, 2, 1, 1)
            
            # Add groupbox to list to store
            self.directory_settings_groupboxes.append(groupbox)
        main_groupbox.setLayout(main_groupbox_layout)
        
        self.directory_settings_layout.addWidget(main_groupbox)    
        # Set layout to the widget
        self.directory_settings.setLayout(self.directory_settings_layout)
        
    def openFileDialogToSetAttribute(self, dir_path, name, attribute, line_edit_widget):
        # Open the file dialog
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                            f'Choose folder for {name}',
                                                            dir_path
                                                            )
        # Enter directory name in display bar
        setattr(self, attribute, folder)
        
        # Edit corresponding widget
        line_edit_widget.setText(folder)
        
    def showSettings(self):
        self.settings_widget.updatePosition()      
        self.settings_widget.show()
        
    def closeEvent(self, event):
        if window.setup:
            result = QtWidgets.QMessageBox.question(self,
                          "Confirm Exit...",
                          "Are you sure you want to exit ?",
                          QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            event.ignore()
    
            if result == QtWidgets.QMessageBox.Yes:
                # Close the plots upon exiting
                self.mainPlot.close()
                self.vtkWidgetViewer.Finalize()
                logging.getLogger().removeHandler(self.logTextBox)
                
                # Close the main window
                event.accept()
        else:
            event.accept()
            
    def setStyleSheet(self):
        self.setStyleSheet("""
            QMenuBar {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
                border: 1px solid #000;
            }
    
            QMenuBar::item {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
            }
    
            QMenuBar::item::selected {
                background-color: rgb(30,30,30);
            }
    
            QMenu {
                background-color: rgb(49,49,49);
                color: rgb(255,255,255);
                border: 1px solid #000;           
            }
    
            QMenu::item::selected {
                background-color: rgb(30,30,30);
            }
        """)
            
    def CreateMenuBar(self):
        # Add the menu bar
        self.widgetMenuBar = self.menuBar()
        
        ### Add menu for "Action"
        self.widgetMenuAction = self.widgetMenuBar.addMenu("&Action")
        # Add action for Action to load in test values
        self.widgetMenuActionSetTestDataBins = self.widgetMenuAction.addAction("Quick-Set test data - Bins + cube")
        self.widgetMenuActionSetTestDataBins.triggered.connect(self.setTestDataBinsCube)
        self.widgetMenuActionSetTestDataTracers = self.widgetMenuAction.addAction("Quick-Set test data - Tracers + cube")
        self.widgetMenuActionSetTestDataTracers.triggered.connect(self.setTestDataTracersCube)
        
        # Add separator
        self.widgetMenuAction.addSeparator()
        
        # Add "Take picture" action
        self.widgetMenuActionTakePicture = self.widgetMenuAction.addAction('Take picture')
        self.widgetMenuActionTakePicture.setShortcut(QtGui.QKeySequence("Ctrl+P"))
        self.widgetMenuActionTakePicture.triggered.connect(self.actionTakePicture)
        
        ### Add menu for "Preferences"
        self.widgetMenuPreferences = self.widgetMenuBar.addMenu("&Preferences")
        # Add action to open settings
        self.widgetMenuPreferencesOpenSettings = self.widgetMenuPreferences.addAction('Settings')
        self.widgetMenuPreferencesOpenSettings.triggered.connect(self.showSettings)
        
    def setTestDataTracersCube(self):
        #####################################################################
        # Change view to the control tab
        self.controllerTabsWidget.setCurrentIndex(1)
        
        #####################################################################
        # Set STL data
        self.modelPDFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Erik_cube_binning_32x32x32\Cube_12cm.stl"
        # Enter filename in display bar
        self.loadModelPDDisplayFile.setText(self.modelPDFileName)
        
        # Set checkbox for ground plane
        self.groundPlaneCheckBox.setChecked(True)
        
        # Set checkbox for show model stl
        self.modelPDShowCheckButton.setChecked(True)
        
        # Load model STL
        self.actionLoadModelPD()
        
        #####################################################################
        # Set outputmesh data
        # Open file dialog
        self.outputMeshFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes\cube\cube-surface-similar_mesh_size_as_binning2.vtk"
        # Enter filename in display bar
        self.loadOutputMeshDisplayFile.setText(self.outputMeshFileName)
        
        # Set checkbox for showing outputmesh
        self.outputMeshShowCheckButton.setChecked(True)
        
        # Load outputmesh VTK file
        self.actionLoadOutputMesh()
        
        #####################################################################
        # Set fluid data
        # Pre-determine the folder to open
        self.fluidDataTypeTracerRadioButton.setChecked(True)
            
        # Open file dialog
        self.fluidDataFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax200_ymin-200_ymax200_zmin-20_zmax200-Filter1.npy"
        # Enter filename in display bar
        self.loadFluidDataDisplayFile.setText(self.fluidDataFileName)
        
        # Set display settings
        self.fluidDataMaskShowCheckButton.setChecked(False)
        self.fluidDataViewControlPercentageSlider.setValue(1)
        self.actionUpdateFluidDataSliderLabel()
        self.fluidDataMaskPDCheckButton.setChecked(True)
        self.fluidDataMaskValidCheckButton.setChecked(True)
        self.fluidDataMaskCropCheckButton.setChecked(True)
        
        # Load the fluiddata
        self.actionLoadFluidDataThreaded()
        
        #####################################################################
        # Set tracer data
        # Open file dialog
        self.tracerDataGFFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax200_ymin-200_ymax200_zmin-20_zmax200-Filter1.npy"
        # Enter filename in display bar
        self.loadTracerDataGFDisplayFile.setText(self.tracerDataGFFileName)
        
        # Set check box for showing tracer data
        self.tracerDataShowCheckBox.setChecked(False)
        
        # Load tracer data file
        self.actionLoadTracerDataGFThreaded()
        
        #####################################################################
        # Set run settings
        self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth.setChecked(True)
        self.mainSettingsRadiusText.setText('10')
        self.updateRadiusSizeLabel()
        
        self.mainSettingsCoinHeightText.setText('3')
        self.updateCoinHeightLabel()
        
        self.mainSettingsCoinOverlapValue.setValue(50)
        self.mainSettingsCoinFitCheckBoxConstrain.setChecked(False)
        self.mainSettingsCoinFitButtonLinear.setChecked(True)
        
        # Set ground plane folder
        self.groundPlaneFitFolder = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\GroundPlaneFits'
        # Enter directory name in display bar
        self.groundPlaneFitFolderLine.setText(self.groundPlaneFitFolder)
        # Set checkbox
        self.showGroundPlaneCorrectedCheckButton.setChecked(True)
        
        
        
    def setTestDataBinsCube(self):
        #####################################################################
        # Change view to the control tab
        self.controllerTabsWidget.setCurrentIndex(1)
        
        #####################################################################
        # Set STL data
        self.modelPDFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Erik_cube_binning_32x32x32\Cube_12cm.stl"
        # Enter filename in display bar
        self.loadModelPDDisplayFile.setText(self.modelPDFileName)
        
        # Set checkbox for ground plane
        self.groundPlaneCheckBox.setChecked(True)
        
        # Set checkbox for show model stl
        self.modelPDShowCheckButton.setChecked(True)
        
        # Load model STL
        self.actionLoadModelPD()
        
        #####################################################################
        # Set outputmesh data
        # Open file dialog
        self.outputMeshFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes\cube\cube-surface-similar_mesh_size_as_binning2.vtk"
        # Enter filename in display bar
        self.loadOutputMeshDisplayFile.setText(self.outputMeshFileName)
        
        # Set checkbox for showing outputmesh
        self.outputMeshShowCheckButton.setChecked(True)
        
        # Load outputmesh VTK file
        self.actionLoadOutputMesh()
        
        #####################################################################
        # Set fluid data
        # Pre-determine the folder to open
        self.fluidDataTypeBinRadioButton.setChecked(True)
            
        # Open file dialog
        self.fluidDataFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Bins\Cube_binning_32x32x32_75per0001.dat"
        # Enter filename in display bar
        self.loadFluidDataDisplayFile.setText(self.fluidDataFileName)
        
        # Set display settings
        self.fluidDataMaskShowCheckButton.setChecked(False)
        self.fluidDataViewControlPercentageSlider.setValue(1)
        self.actionUpdateFluidDataSliderLabel()
        self.fluidDataMaskPDCheckButton.setChecked(True)
        self.fluidDataMaskValidCheckButton.setChecked(True)
        self.fluidDataMaskCropCheckButton.setChecked(True)
        
        # Load the fluiddata
        self.actionLoadFluidDataThreaded()
        
        #####################################################################
        # Set tracer data
        # Open file dialog
        self.tracerDataGFFileName = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax200_ymin-200_ymax200_zmin-20_zmax200-Filter1.npy"
        # Enter filename in display bar
        self.loadTracerDataGFDisplayFile.setText(self.tracerDataGFFileName)
        
        # Set check box for showing tracer data
        self.tracerDataShowCheckBox.setChecked(False)
        
        # Load tracer data file
        self.actionLoadTracerDataGFThreaded()
        
        #####################################################################
        # Set run settings
        self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp.setChecked(True)
        self.mainSettingsRadiusText.setText('3 * $pitchMM')
        self.updateRadiusSizeLabel()
        
        # Set ground plane folder
        self.groundPlaneFitFolder = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\GroundPlaneFits'
        # Enter directory name in display bar
        self.groundPlaneFitFolderLine.setText(self.groundPlaneFitFolder)
        # Set checkbox
        self.showGroundPlaneCorrectedCheckButton.setChecked(True)

    def CreateViewerTabsWidget(self):
        # Create the tab widget
        self.viewerTabsWidget = QtWidgets.QTabWidget()
        
        # Create the controller widget which will lie inside the tabs widget
        self.CreateVTKWidgetViewer()
        self.CreatePlotViewer()
        
        # Add the VTK controller widget to the viewerTabsWidget
        self.viewerTabsWidget.addTab(self.vtkWidgetViewer, 'Three-dimensional viewer')
        self.viewerTabsWidget.addTab(self.mainPlotViewer, 'Plot viewer')
        
        # Finally redefine minimum size
        # self.tabsWidget.setMinimumSize(200, 200)
    
    def CreateControllerTabsWidget(self):
        # Create the tab widget
        # self.controllerTabsWidget = QtTabWidgetCollapsible()
        self.controllerTabsWidget = QtTabWidgetCollapsible()
        
        # Create the controller widget which will lie inside the tabs widget
        self.CreateVTKWidgetController()
        self.CreateMainSettingsController()
        
        self.emptyWidget = QtWidgets.QWidget()
        # Add the VTK controller widget to the controllerTabsWidget
        
        # Add geometry tabs
        self.controllerTabsWidget.addTab(self.controllerWidgetGeometry, 'Geometry objects')
        self.controllerTabsWidget.addTab(self.controllerWidgetFluid, 'Fluid objects')
        self.controllerTabsWidget.addTab(self.mainSettingsControllerWidget, 'Run settings')
        # self.controllerTabsWidget.addTab(self.emptyWidget, '\/')
        # self.controllerTabLayout.addWidget(self.controllerWidget, 'Mesh definitions')
        # self.controllerTabLayout.addWidget(self.mainSettingsControllerWidget, 'Run settings')
        
        
        # Finally redefine minimum size
        # self.controllerTabsWidget.setMinimumWidth(200)
        self.controllerTabsWidget.defMaximumWidth(550)
    
    def CreatePlotViewer(self):
        # Create the widget that holds the plots
        self.mainPlotViewer = QtWidgets.QWidget()
        self.mainPlotViewerLayout = QtWidgets.QVBoxLayout()
        
        # Create a multi-plot widget using pyvista
        shape = (4, 4)
        row_weights = [1, 1, 1, 1]#0.667]
        col_weights = [1, 1, 1, 1]#0.667]
        
        groups = [
            ([0, 2], [0, 2]),
            # (3, [0, 2]),
            # ([0, 2], 3)
            ]
        
        self.mainPlot = QtInteractor(self.mainPlotViewer, shape=shape,
                                     row_weights=row_weights,
                                     col_weights = col_weights,
                                     groups=groups)
        self.mainPlotViewerLayout.addWidget(self.mainPlot.interactor)
        self.mainPlotViewer.setLayout(self.mainPlotViewerLayout)
        
        self.mainPlotSubplotsList = np.ndarray((7,),dtype=object)
        
        # Predefine plots (same as clearing)
        self.resetPlots()
        
        self.chartDataLoaded = False
        
    def sliceSphereChoiceUpdated(self, val):
        self.mainSettingsSliceSphereRadiusGroupBox.setEnabled(val)
        self.mainSettingsSliceSphereRadiusUsePlane.setEnabled(val)
        
        
    def CreateMainSettingsController(self):
        # Create the main widget which holds all settings
        self.mainSettingsControllerWidget = QtWidgets.QWidget()
        # Create a VBox layout
        self.mainSettingsControllerWidgetLayout = QtWidgets.QVBoxLayout()
        
        # Split settings in two groupboxes
        self.mainSettingsSettingsGroupBox = QtWidgets.QGroupBox('Main settings')
        self.mainSettingsSettingsGroupBoxLayout = QtWidgets.QVBoxLayout()
        # self.mainSettingsRunGroupBox = QtWidgets.QGroupBox('Evaluate algorithm')
        # self.mainSettingsRunGroupBoxLayout = QtWidgets.QVBoxLayout()
        
        ##############
        # Add a groupbox to hold the chosen setting
        self.mainSettingsMethodNumberGroupBox = QtWidgets.QGroupBox('Choose Algorithm')
        self.mainSettingsMethodNumberGroupBoxLayout = QtWidgets.QGridLayout()
        # Add button for Jux' method
        self.mainSettingsMethodNumberGroupBoxButtonJux = QtWidgets.QRadioButton("Jux' linear extraction")
        # Add button for Bin-Con-LinInterp method
        self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp = QtWidgets.QRadioButton("Bin-Con-LinInterp")
        # Add button for Bin-Nudge-QuadFit method
        self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit = QtWidgets.QRadioButton("Bin-Nudge-QuadFit")
        # Add button for Tracer-Nudge-QuadFit method
        self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit = QtWidgets.QRadioButton("Tracer-Nudge-QuadFit")
        # Add button for Tracer-Con-QuadFit method
        self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit = QtWidgets.QRadioButton("Tracer-Con-QuadFit")
        # Add button for Tracer-Con-QuadFit method
        self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth = QtWidgets.QRadioButton("Coin-Based-Tracer-LinReg")
        
        # Create a button group to hold all buttons
        self.mainSettingsMethodNumberGroupBoxButtons = QtWidgets.QButtonGroup()
        self.mainSettingsMethodNumberGroupBoxButtons.addButton(self.mainSettingsMethodNumberGroupBoxButtonJux, 0)
        self.mainSettingsMethodNumberGroupBoxButtons.addButton(self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp, 1)
        self.mainSettingsMethodNumberGroupBoxButtons.addButton(self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit, 2)
        self.mainSettingsMethodNumberGroupBoxButtons.addButton(self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit, 3)
        self.mainSettingsMethodNumberGroupBoxButtons.addButton(self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit, 4)
        self.mainSettingsMethodNumberGroupBoxButtons.addButton(self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth, 5)
        
        # Add buttons to groupbox
        self.mainSettingsMethodNumberGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBoxButtonJux, 0, 0, 1, 1)
        self.mainSettingsMethodNumberGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp, 1, 0, 1, 1)
        self.mainSettingsMethodNumberGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit, 2, 0, 1, 1)
        self.mainSettingsMethodNumberGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit, 0, 1, 1, 1)
        self.mainSettingsMethodNumberGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit, 1, 1, 1, 1)
        self.mainSettingsMethodNumberGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth, 2, 1, 1, 1)
        
        # Set the layout
        self.mainSettingsMethodNumberGroupBox.setLayout(self.mainSettingsMethodNumberGroupBoxLayout)
        
        self.mainSettingsMethodNumberGroupBoxButtonJux.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth.setEnabled(False)
        
        ##############
        # Add a tab for the main settings and 
        self.mainSettingsConfigTabHolder = QtWidgets.QTabWidget()
        
        # Add a tab with general settings for all methods
        # self.mainSettingsConfigGroupBox = QtWidgets.QGroupBox('Configurate run settings')
        # self.mainSettingsConfigGroupBoxLayout = QtWidgets.QGridLayout()
        self.mainSettingsConfigTab = QtWidgets.QWidget()
        self.mainSettingsConfigTabLayout = QtWidgets.QGridLayout()
        
        # Add a tab with settings for the coin-based method
        self.mainSettingsConfigCoinBasedTab = QtWidgets.QWidget()
        self.mainSettingsConfigCoinBasedTabLayout = QtWidgets.QGridLayout()
        
        self.mainSettingsConfigTabHolder.addTab(self.mainSettingsConfigTab, 'Configurate run settings')
        self.mainSettingsConfigTabHolder.addTab(self.mainSettingsConfigCoinBasedTab, 'Configurate coin-based settings')
        # self.mainSettingsConfigCoinBasedTab.setEnabled(False)
        
# =============================================================================
#         ## Create the sphere settings
# =============================================================================
        # Create a line edit box
        self.mainSettingsRadiusText = QtWidgets.QLineEdit()
        self.mainSettingsRadiusText.textChanged.connect(self.updateRadiusSizeLabel)
        self.mainSettingsRadiusLabel = QtWidgets.QLabel()
        self.mainSettingsRadiusLabel.setText('Interrogation sphere radius [mm]')
        
        # Create a label that shows the result
        self.mainSettingsRadiusValue = QtWidgets.QLabel()
        self.mainSettingsRadiusValue.setText('= ... mm')
        
        # Slice sphere box
        self.mainSettingsSliceOnCheckbox = QtWidgets.QCheckBox('Slice sphere')
        self.mainSettingsSliceOnCheckbox.stateChanged.connect(self.sliceSphereChoiceUpdated)
        
        # Add spin box for sphere radius
        self.mainSettingsSliceSphereRadiusSpinBox = QtWidgets.QSpinBox()
        self.mainSettingsSliceSphereRadiusSpinBox.setSuffix(' mm')
        self.mainSettingsSliceSphereRadiusSpinBox.setMinimum(0)
        self.mainSettingsSliceSphereRadiusSpinBox.setMaximum(1000)
        self.mainSettingsSliceSphereRadiusLabel = QtWidgets.QLabel('Radius of curvature')
        self.mainSettingsSliceSphereRadiusGroupBox = QtWidgets.QGroupBox()
        self.mainSettingsSliceSphereRadiusGroupBoxLayout = QtWidgets.QHBoxLayout()
        self.mainSettingsSliceSphereRadiusGroupBoxLayout.addWidget(self.mainSettingsSliceSphereRadiusLabel)
        self.mainSettingsSliceSphereRadiusGroupBoxLayout.addWidget(self.mainSettingsSliceSphereRadiusSpinBox)
        self.mainSettingsSliceSphereRadiusGroupBox.setLayout(self.mainSettingsSliceSphereRadiusGroupBoxLayout)
        
        # Use planar slice check box
        self.mainSettingsSliceSphereRadiusUsePlane = QtWidgets.QCheckBox('Use planar slice')
        self.mainSettingsSliceGroupBox = QtWidgets.QGroupBox()
        self.mainSettingsSliceSphereRadiusUsePlane.stateChanged.connect(lambda val: self.mainSettingsSliceSphereRadiusGroupBox.setEnabled(not val))
        self.mainSettingsSliceGroupBoxLayout = QtWidgets.QHBoxLayout()
        
        self.mainSettingsSliceGroupBoxLayout.addWidget(self.mainSettingsSliceOnCheckbox)
        self.mainSettingsSliceGroupBoxLayout.addWidget(self.mainSettingsSliceSphereRadiusGroupBox)
        self.mainSettingsSliceGroupBoxLayout.addWidget(self.mainSettingsSliceSphereRadiusUsePlane)
        
        
        self.mainSettingsSliceGroupBox.setFlat(True)
        # This removes the border from a QGroupBox named "theBox".
        self.mainSettingsSliceGroupBox.setStyleSheet("QGroupBox#self.mainSettingsSliceGroupBox {border:0;}")
        
        self.mainSettingsSliceGroupBox.setLayout(self.mainSettingsSliceGroupBoxLayout)
        
        self.sliceSphereChoiceUpdated(False)
        
# =============================================================================
#         ## Create the coin settings
# =============================================================================
        # Create a line edit box
        self.mainSettingsCoinHeightText = QtWidgets.QLineEdit()
        self.mainSettingsCoinHeightText.textChanged.connect(self.updateCoinHeightLabel)
        self.mainSettingsCoinHeightLabel = QtWidgets.QLabel()
        self.mainSettingsCoinHeightLabel.setText('Interrogation coin height [mm]')
        
        # Create a label that shows the result
        self.mainSettingsCoinHeightValue = QtWidgets.QLabel()
        self.mainSettingsCoinHeightValue.setText('= ... mm')
        
        # Create the overlap
        self.mainSettingsCoinOverlapLabel = QtWidgets.QLabel()
        self.mainSettingsCoinOverlapLabel.setText('Coin overlap')
        
        self.mainSettingsCoinOverlapValue = QtWidgets.QSpinBox()
        self.mainSettingsCoinOverlapValue.setMinimum(0)
        self.mainSettingsCoinOverlapValue.setMaximum(99)
        self.mainSettingsCoinOverlapValue.setSuffix(' %')
        
        ## Add a group box for the velocity fitting methods inside the coin
        self.mainSettingsCoinFitMethodGroupBox = QtWidgets.QGroupBox('Coin-Velocity fit method')
        self.mainSettingsCoinFitMethodGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Add buttons for all methods
        self.mainSettingsCoinFitButtonLinear = QtWidgets.QRadioButton('Linear')
        self.mainSettingsCoinFitButtonQuadratic = QtWidgets.QRadioButton('Quadratic')
        self.mainSettingsCoinFitButtonCubic = QtWidgets.QRadioButton('Cubic')
        self.mainSettingsCoinFitButtonClauser = QtWidgets.QRadioButton('Clauser')
        
        # Add all methods to a buttongroup
        self.mainSettingsCoinFitButtonGroup = QtWidgets.QButtonGroup()
        self.mainSettingsCoinFitButtonGroup.addButton(self.mainSettingsCoinFitButtonLinear, 0)
        # self.mainSettingsCoinFitButtonGroup.addButton(self.mainSettingsCoinFitButtonQuadratic, 1)
        # self.mainSettingsCoinFitButtonGroup.addButton(self.mainSettingsCoinFitButtonCubic, 2)
        # self.mainSettingsCoinFitButtonGroup.addButton(self.mainSettingsCoinFitButtonClauser, 3)
        
        # Add all methods to the groupbox
        self.mainSettingsCoinFitMethodGroupBoxLayout.addWidget(self.mainSettingsCoinFitButtonLinear, 0, 0, 1, 1)
        # self.mainSettingsCoinFitMethodGroupBoxLayout.addWidget(self.mainSettingsCoinFitButtonQuadratic, 1, 0, 1, 1)
        # self.mainSettingsCoinFitMethodGroupBoxLayout.addWidget(self.mainSettingsCoinFitButtonCubic, 0, 1, 1, 1)
        # self.mainSettingsCoinFitMethodGroupBoxLayout.addWidget(self.mainSettingsCoinFitButtonClauser, 1, 1, 1, 1)
        # 
        self.mainSettingsCoinFitMethodGroupBox.setLayout(self.mainSettingsCoinFitMethodGroupBoxLayout)
        
        # Add a checkbox for setting the constraint at the object
        self.mainSettingsCoinFitCheckBoxConstrain = QtWidgets.QCheckBox('Constrain velocity at wall')
# =============================================================================
#         #### Do the cropping dimensions
# =============================================================================

        # Create the sliders and up down boxes
        self.rangeCropTracersGroupbox = QtWidgets.QGroupBox('Crop dimensions - Ground Estimate')
        self.rangeCropTracersGroupboxLayout = QtWidgets.QGridLayout()
        
        # Create labels
        self.rangeCropTracersMinLabel = QtWidgets.QLabel('Min:')
        self.rangeCropTracersMaxLabel = QtWidgets.QLabel('Max:')
        
        ### Create horizontal sliders (factor 5)
        self.rangeCropTracersMinSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rangeCropTracersMinSlider.setMinimum(-10)
        self.rangeCropTracersMinSlider.setMaximum(20)
        self.rangeCropTracersMinSlider.setValue(-1)
        self.rangeCropTracersMinSlider.setTickInterval(5)
        self.rangeCropTracersMinSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.rangeCropTracersMinSlider.valueChanged.connect(self.actionCropTracerMinSliderChanged)
        
        self.rangeCropTracersMaxSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rangeCropTracersMaxSlider.setMinimum(-10)
        self.rangeCropTracersMaxSlider.setMaximum(20)
        self.rangeCropTracersMaxSlider.setValue(2)
        self.rangeCropTracersMaxSlider.setTickInterval(5)
        self.rangeCropTracersMaxSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rangeCropTracersMaxSlider.valueChanged.connect(self.actionCropTracerMaxSliderChanged)
        
        # Create spinboxes
        self.rangeCropTracersMin = QtWidgets.QSpinBox()
        self.rangeCropTracersMin.setMinimum(-50)
        self.rangeCropTracersMin.setMaximum(100)
        self.rangeCropTracersMin.setValue(-5)
        self.rangeCropTracersMin.setSingleStep(5)
        self.rangeCropTracersMin.setSuffix(" mm")
        self.rangeCropTracersMin.valueChanged.connect(self.actionCropTracerMinChanged)
        
        self.rangeCropTracersMax = QtWidgets.QSpinBox()
        self.rangeCropTracersMax.setMinimum(-50)
        self.rangeCropTracersMax.setMaximum(100)
        self.rangeCropTracersMax.setValue(10)
        self.rangeCropTracersMax.setSingleStep(5)
        self.rangeCropTracersMax.setSuffix(" mm")
        self.rangeCropTracersMax.valueChanged.connect(self.actionCropTracerMaxChanged)
        
        # Add to groupbox layout
        self.rangeCropTracersGroupboxLayout.addWidget(self.rangeCropTracersMinLabel, 0, 0, 1, 1)
        self.rangeCropTracersGroupboxLayout.addWidget(self.rangeCropTracersMinSlider, 0, 1, 1, 2)
        self.rangeCropTracersGroupboxLayout.addWidget(self.rangeCropTracersMin, 0, 3, 1, 2)
        self.rangeCropTracersGroupboxLayout.addWidget(self.rangeCropTracersMaxLabel, 1, 0, 1, 1)
        self.rangeCropTracersGroupboxLayout.addWidget(self.rangeCropTracersMaxSlider, 1, 1, 1, 2)
        self.rangeCropTracersGroupboxLayout.addWidget(self.rangeCropTracersMax, 1, 3, 1, 2)
        # Set to layout
        self.rangeCropTracersGroupbox.setLayout(self.rangeCropTracersGroupboxLayout)
        
        ####################
        
        # Create the sliders and up down boxes
        self.rangeCropVelfieldGroupbox = QtWidgets.QGroupBox('Crop dimensions - Fluid Mesh')
        self.rangeCropVelfieldGroupboxLayout = QtWidgets.QGridLayout()
        
        # Create labels
        self.rangeCropVelfieldMinLabel = QtWidgets.QLabel('Min:')
        self.rangeCropVelfieldMaxLabel = QtWidgets.QLabel('Max:')
        
        ### Create horizontal sliders (factor 5)
        self.rangeCropVelfieldMinSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rangeCropVelfieldMinSlider.setMinimum(-10)
        self.rangeCropVelfieldMinSlider.setMaximum(20)
        self.rangeCropVelfieldMinSlider.setValue(-4)
        self.rangeCropVelfieldMinSlider.setTickInterval(5)
        self.rangeCropVelfieldMinSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.rangeCropVelfieldMinSlider.valueChanged.connect(self.actionCropVelfieldMinSliderChanged)
        
        self.rangeCropVelfieldMaxSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rangeCropVelfieldMaxSlider.setMinimum(-10)
        self.rangeCropVelfieldMaxSlider.setMaximum(20)
        self.rangeCropVelfieldMaxSlider.setValue(10)
        self.rangeCropVelfieldMaxSlider.setTickInterval(5)
        self.rangeCropVelfieldMaxSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rangeCropVelfieldMaxSlider.valueChanged.connect(self.actionCropVelfieldMaxSliderChanged)
        
        # Create spinboxes
        self.rangeCropVelfieldMin = QtWidgets.QSpinBox()
        self.rangeCropVelfieldMin.setMinimum(-50)
        self.rangeCropVelfieldMin.setMaximum(100)
        self.rangeCropVelfieldMin.setValue(-20)
        self.rangeCropVelfieldMin.setSingleStep(5)
        self.rangeCropVelfieldMin.setSuffix(" mm")
        self.rangeCropVelfieldMin.valueChanged.connect(self.actionCropVelfieldMinChanged)
        
        self.rangeCropVelfieldMax = QtWidgets.QSpinBox()
        self.rangeCropVelfieldMax.setMinimum(-50)
        self.rangeCropVelfieldMax.setMaximum(100)
        self.rangeCropVelfieldMax.setValue(50)
        self.rangeCropVelfieldMax.setSingleStep(5)
        self.rangeCropVelfieldMax.setSuffix(" mm")
        self.rangeCropVelfieldMax.valueChanged.connect(self.actionCropVelfieldMaxChanged)
        
        # Add to groupbox layout
        self.rangeCropVelfieldGroupboxLayout.addWidget(self.rangeCropVelfieldMinLabel, 0, 0, 1, 1)
        self.rangeCropVelfieldGroupboxLayout.addWidget(self.rangeCropVelfieldMinSlider, 0, 1, 1, 2)
        self.rangeCropVelfieldGroupboxLayout.addWidget(self.rangeCropVelfieldMin, 0, 3, 1, 2)
        self.rangeCropVelfieldGroupboxLayout.addWidget(self.rangeCropVelfieldMaxLabel, 1, 0, 1, 1)
        self.rangeCropVelfieldGroupboxLayout.addWidget(self.rangeCropVelfieldMaxSlider, 1, 1, 1, 2)
        self.rangeCropVelfieldGroupboxLayout.addWidget(self.rangeCropVelfieldMax, 1, 3, 1, 2)
        # Set to layout
        self.rangeCropVelfieldGroupbox.setLayout(self.rangeCropVelfieldGroupboxLayout)
        
        # Add start button
        self.startSetupButton = QtWidgets.QPushButton('Start Setup')
        self.startSetupButton.clicked.connect(self.actionRunSetup)
        
        self.mainSettingsSettingsGroupBoxLayout.addWidget(self.mainSettingsMethodNumberGroupBox)
        self.mainSettingsSettingsGroupBoxLayout.addWidget(self.mainSettingsConfigTabHolder)
        self.mainSettingsSettingsGroupBoxLayout.addWidget(self.startSetupButton)
        self.mainSettingsSettingsGroupBox.setLayout(self.mainSettingsSettingsGroupBoxLayout)
# =============================================================================
#         # Create small segment for running single interrogation sphere iteration
# =============================================================================
        # Create large groupbox for evaluating
        self.evaluationGroupBox = QtWidgets.QGroupBox('Run evaluation')
        self.evaluationGroupBoxLayout = QtWidgets.QVBoxLayout()
        
        #####################################################
        # Create settings groupbox for evaluation
        self.evaluationSettingsGroupBox = QtWidgets.QGroupBox('Evaluation settings')
        self.evaluationSettingsGroupBoxLayout = QtWidgets.QGridLayout()
        
        # findBL
        self.evaluationExtractBLParametersLabel = QtWidgets.QCheckBox('Extract Boundary Layer Parameters')
        
        # Save file
        self.evaluationSaveFilePath = QtWidgets.QLineEdit()
        
        # Min distance for 
        
        # Create ground plane fitting box
        self.groundPlaneEvaluationGroupBox = QtWidgets.QGroupBox('Ground plane fitting')
        self.groundPlaneEvaluationGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Create checkbox
        self.showGroundPlaneCorrectedCheckButton = QtWidgets.QCheckBox('Show corrected ground plane fit in 3D view')
        self.showGroundPlaneCorrectedCheckButton.stateChanged.connect(self.showGroundPlaneCorrected)
        
        # Create place to save folder info
        self.groundPlaneFitFolderLine = QtWidgets.QLineEdit()
        self.groundPlaneFitFolderLine.setReadOnly(True)
        self.groundPlaneFitFolderOpenFileDialog = QtWidgets.QPushButton('Choose ground-plane fit folder')
        self.groundPlaneFitFolderOpenFileDialog.clicked.connect(self.actionOpenGroundPlaneFitFolderDialog)
        
        # Create pushbutton for ground plane fit
        self.groundPlaneEvaluationButton = QtWidgets.QPushButton('Execute ground plane fit')
        self.groundPlaneEvaluationButton.clicked.connect(self.actionFitGroundPlane)
        
        # Add widgets to the layout
        self.groundPlaneEvaluationGroupBoxLayout.addWidget(self.showGroundPlaneCorrectedCheckButton, 0, 0, 1, -1)
        self.groundPlaneEvaluationGroupBoxLayout.addWidget(self.groundPlaneFitFolderLine, 1, 0, 1, 1)
        self.groundPlaneEvaluationGroupBoxLayout.addWidget(self.groundPlaneFitFolderOpenFileDialog, 1, 1, 1, 1)
        self.groundPlaneEvaluationGroupBoxLayout.addWidget(self.groundPlaneEvaluationButton, 2, 0, 1, -1)
        
        self.groundPlaneEvaluationGroupBox.setLayout(self.groundPlaneEvaluationGroupBoxLayout)
        
        #####################################################
        # Create groupbox for SINGLE iterate
        self.singleEvaluationGroupBox = QtWidgets.QGroupBox('Single iterate')
        self.singleEvaluationGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Add a combo box for the number
        self.singleEvaluationIdxLabel = QtWidgets.QLabel()
        self.singleEvaluationIdxLabel.setText('Iterate index')
        
        self.singleEvaluationIdx = QtWidgets.QSpinBox()
        self.singleEvaluationIdx.setValue(0)
        self.singleEvaluationIdx.setMinimum(0)
        self.singleEvaluationIdx.valueChanged.connect(self.highlightCellInOutputMesh)
        
        # Add button to perform single evaluation
        self.singleEvaluationButton = QtWidgets.QPushButton('Evaluate single iterate')
        self.singleEvaluationButton.pressed.connect(self.actionEvaluateSingleIterate)
        
        # Add widgets to the layout
        self.singleEvaluationGroupBoxLayout.addWidget(self.singleEvaluationIdxLabel, 0, 0, 1, 1)
        self.singleEvaluationGroupBoxLayout.addWidget(self.singleEvaluationIdx, 0, 1, 1, 1)
        self.singleEvaluationGroupBoxLayout.addWidget(self.singleEvaluationButton, 1, 0, 1, -1)
        
        # Set layout to small box
        self.singleEvaluationGroupBox.setLayout(self.singleEvaluationGroupBoxLayout)
        
        #####################################################
        # Create groupbox for WHOLE iterate
        self.fullEvaluationGroupBox = QtWidgets.QGroupBox('Run whole iterate')
        self.fullEvaluationGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Add line edit to save-directory
        self.saveDirectoryLine = QtWidgets.QLineEdit()
        self.saveDirectoryLine.setReadOnly(True)
        self.saveDirectoryOpenFileDialog = QtWidgets.QPushButton('Choose save-folder')
        self.saveDirectoryOpenFileDialog.clicked.connect(self.actionOpenSaveDirectoryDialog)
        
        # Add button to start full evaluation
        self.fullEvaluationButtonStart = QtWidgets.QPushButton('START')
        self.fullEvaluationButtonStop = QtWidgets.QPushButton('STOP')
        self.fullEvaluationButtonStart.pressed.connect(self.actionRunMain)
        self.fullEvaluationButtonStop.pressed.connect(self.actionStopMainRun)
        
        # Add start and stop to single groupbox
        self.fullEvaluationButtonGroupBox = QtWidgets.QGroupBox()
        self.fullEvaluationButtonGroupBoxLayout = QtWidgets.QHBoxLayout()
        self.fullEvaluationButtonGroupBoxLayout.addWidget(self.fullEvaluationButtonStart)
        self.fullEvaluationButtonGroupBoxLayout.addWidget(self.fullEvaluationButtonStop)
        self.fullEvaluationButtonGroupBox.setLayout(self.fullEvaluationButtonGroupBoxLayout)
        
        self.fullEvaluationButtonGroupBox.setFlat(True)
        self.fullEvaluationButtonGroupBox.setStyleSheet("QGroupBox {border:0;}")
        
        self.fullEvaluationGroupBoxLayout.addWidget(self.saveDirectoryLine, 0, 0, 1, 1)
        self.fullEvaluationGroupBoxLayout.addWidget(self.saveDirectoryOpenFileDialog, 0, 1, 1, 1)
        self.fullEvaluationGroupBoxLayout.addWidget(self.fullEvaluationButtonGroupBox, 1, 0, 1, -1)
        
        self.fullEvaluationGroupBox.setLayout(self.fullEvaluationGroupBoxLayout)
        
        ##########################
        # Add groupbox widgets to the large box
        self.evaluationGroupBoxLayout.addWidget(self.groundPlaneEvaluationGroupBox)
        self.evaluationGroupBoxLayout.addWidget(self.singleEvaluationGroupBox)
        self.evaluationGroupBoxLayout.addWidget(self.fullEvaluationGroupBox)
        
        self.evaluationGroupBox.setLayout(self.evaluationGroupBoxLayout)
        self.evaluationGroupBox.setEnabled(False)


# =============================================================================
#         # Add everything to the layout
# =============================================================================
        # Heneral settings
        self.mainSettingsConfigTabLayout.addWidget(self.mainSettingsRadiusLabel, 0, 0, 1, 1)
        self.mainSettingsConfigTabLayout.addWidget(self.mainSettingsRadiusText, 0, 1, 1, 1)
        self.mainSettingsConfigTabLayout.addWidget(self.mainSettingsRadiusValue, 0, 2, 1, 1)
        self.mainSettingsConfigTabLayout.addWidget(self.mainSettingsSliceGroupBox, 1, 0, 1, -1)
        self.mainSettingsConfigTabLayout.addWidget(self.rangeCropTracersGroupbox, 2, 0, 1, -1)
        self.mainSettingsConfigTabLayout.addWidget(self.rangeCropVelfieldGroupbox, 3, 0, 1, -1)
        
        self.mainSettingsConfigTab.setLayout(self.mainSettingsConfigTabLayout)
        
        # Coin-specific settings
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinHeightLabel, 0, 0, 1, 1)
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinHeightText, 0, 1, 1, 1)
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinHeightValue, 0, 2, 1, 1)
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinOverlapLabel, 1, 0, 1, 1) # Overlap
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinOverlapValue, 1, 1, 1, 1) # Overlap
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinFitMethodGroupBox, 2, 0, 2, 2) # Fit method
        self.mainSettingsConfigCoinBasedTabLayout.addWidget(self.mainSettingsCoinFitCheckBoxConstrain, 4, 0, 1, -1)
        
        self.mainSettingsConfigCoinBasedTab.setLayout(self.mainSettingsConfigCoinBasedTabLayout)
        
        # Add widget to settings layout
        self.mainSettingsControllerWidgetLayout.addWidget(self.mainSettingsSettingsGroupBox)
        self.mainSettingsControllerWidgetLayout.addWidget(self.evaluationGroupBox)
        
        # Set main settings layout
        self.mainSettingsControllerWidget.setLayout(self.mainSettingsControllerWidgetLayout)
        
    def highlightCellInOutputMesh(self, cell_idx):
        # Check if outputmesh is loaded
        if self.outputMeshLoaded and self.setupIsFinished:
            if self.singleCellPreviouslyLoaded:
                ### Update the cellIdList
                self.cellIdList.SetId(0, cell_idx)
                self.cellExtract.SetCellList(self.cellIdList)
                self.cellExtract.Update()
                
                # Update vtkWidgetRen
                self.vtkWidgetViewer.GetRenderWindow().Render()
                self.vtkWidgetIren.Initialize()
                
            else:
                ### Then highlight the new cell_idx
                # Set opacity to very low
                self.outputMeshActor.GetProperty().SetOpacity(0.3)
                
                # Extract the cell
                self.cellExtract = vtk.vtkExtractCells()
                self.cellExtract.SetInputData(self.vtkOutputMesh)
                self.cellIdList = vtk.vtkIdList()
                self.cellIdList.SetNumberOfIds(1)
                self.cellIdList.SetId(0, cell_idx)
                self.cellExtract.SetCellList(self.cellIdList)
                self.cellExtract.Update()
                
                #########
                # Create a mapper for the cell
                self.cellSurfaceMapper = vtk.vtkDataSetMapper()
                self.cellSurfaceMapper.SetInputConnection(self.cellExtract.GetOutputPort())
                
                # # Create an actor for the cells
                self.cellSurfaceActor = vtk.vtkActor()
                self.cellSurfaceActor.SetMapper(self.cellSurfaceMapper)
                self.cellSurfaceActor.GetProperty().SetRepresentationToSurface()
                self.cellSurfaceActor.GetProperty().SetColor(self.vtkColors.GetColor3d("Crimson"))
                
                #########
                # Create a mapper for the cellPoints
                self.cellPointsMapper = vtk.vtkDataSetMapper()
                self.cellPointsMapper.SetInputConnection(self.cellExtract.GetOutputPort())
                
                # Create an actor for the cellPoints
                self.cellPointsActor = vtk.vtkActor()
                self.cellPointsActor.SetMapper(self.cellPointsMapper)
                self.cellPointsActor.GetProperty().SetRepresentationToPoints()
                self.cellPointsActor.GetProperty().RenderPointsAsSpheresOn()
                self.cellPointsActor.GetProperty().SetColor(self.vtkColors.GetColor3d("Banana"))
                self.cellPointsActor.GetProperty().SetPointSize(10.0)
                
                # Add actor to renderer
                self.vtkWidgetRen.AddActor(self.cellSurfaceActor)
                self.vtkWidgetRen.AddActor(self.cellPointsActor)
                
                # Update vtkWidgetRen
                # self.vtkWidgetRen.ResetCamera()
                self.vtkWidgetViewer.GetRenderWindow().Render()
                self.vtkWidgetIren.Initialize()
                
                self.singleCellPreviouslyLoaded = True
                
    def updateCoinHeightLabel(self):
        # Get string that can be evaluated by replacing "$" with "self."
        hCoinText = self.mainSettingsCoinHeightText.text().replace('$', 'self.')
        
        try:
            if hCoinText == '':
                self.mainSettingsCoinHeightValue.setText('= ... mm')
            else:
                self.HCoin = eval(hCoinText)
                self.mainSettingsCoinHeightValue.setText(f'= {self.HCoin} mm')
        except:
            self.mainSettingsCoinHeightValue.setText('= ??? mm')
    
    def updateRadiusSizeLabel(self):
        # Get string that can be evaluated by replacing "$" with "self."
        rSphereText = self.mainSettingsRadiusText.text().replace('$', 'self.')
        
        try:
            if rSphereText == '':
                self.mainSettingsRadiusValue.setText('= ... mm')
            else:
                self.RSphere = eval(rSphereText)
                self.mainSettingsRadiusValue.setText(f'= {self.RSphere} mm')
        except:
            self.mainSettingsRadiusValue.setText('= ??? mm')
        
        
    @QtCore.pyqtSlot(int)
    def actionCropTracerMinSliderChanged(self, value):
        #1. Update min spinbox
        self.rangeCropTracersMin.setValue(int(value * 5))
        
        #2. If above the set maximum, increase maximum too
        if value > self.rangeCropTracersMaxSlider.value():
            # Set the slider value
            self.rangeCropTracersMaxSlider.setValue(value)
            # Set the spinbox value
            self.rangeCropTracersMax.setValue(int(value * 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropTracerMaxSliderChanged(self, value):
        #1. Update max spinbox
        self.rangeCropTracersMax.setValue(int(value * 5))
        
        #2. If below the set minimum, decrease minimum too
        if value < self.rangeCropTracersMinSlider.value():
            # Set the slider value
            self.rangeCropTracersMinSlider.setValue(value)
            # Set the spinbox value
            self.rangeCropTracersMin.setValue(int(value * 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropTracerMinChanged(self, value):
        #1. Update min slider
        self.rangeCropTracersMinSlider.setValue(int(value / 5))
        
        #2. If above the set maximum, increase maximum too
        if value > self.rangeCropTracersMax.value():
            # Set the spinbox value
            self.rangeCropTracersMax.setValue(value)
            # Set the slider value
            self.rangeCropTracersMaxSlider.setValue(int(value / 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropTracerMaxChanged(self, value):
        #1. Update max slider
        self.rangeCropTracersMaxSlider.setValue(int(value / 5))
        
        #2. If below the set minimum, decrease minimum too
        if value < self.rangeCropTracersMin.value():
            # Set the spinbox value
            self.rangeCropTracersMax.setValue(value)
            # Set the slider value
            self.rangeCropTracersMaxSlider.setValue(int(value / 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropVelfieldMinSliderChanged(self, value):
        #1. Update min spinbox
        self.rangeCropVelfieldMin.setValue(int(value * 5))
        
        #2. If above the set maximum, increase maximum too
        if value > self.rangeCropVelfieldMaxSlider.value():
            # Set the slider value
            self.rangeCropVelfieldMaxSlider.setValue(value)
            # Set the spinbox value
            self.rangeCropVelfieldMax.setValue(int(value * 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropVelfieldMaxSliderChanged(self, value):
        #1. Update max spinbox
        self.rangeCropVelfieldMax.setValue(int(value * 5))
        
        #2. If below the set minimum, decrease minimum too
        if value < self.rangeCropVelfieldMinSlider.value():
            # Set the slider value
            self.rangeCropVelfieldMinSlider.setValue(value)
            # Set the spinbox value
            self.rangeCropVelfieldMin.setValue(int(value * 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropVelfieldMinChanged(self, value):
        #1. Update min slider
        self.rangeCropVelfieldMinSlider.setValue(int(value / 5))
        
        #2. If above the set maximum, increase maximum too
        if value > self.rangeCropVelfieldMax.value():
            # Set the spinbox value
            self.rangeCropVelfieldMax.setValue(value)
            # Set the slider value
            self.rangeCropVelfieldMaxSlider.setValue(int(value / 5))
    
    @QtCore.pyqtSlot(int)
    def actionCropVelfieldMaxChanged(self, value):
        #1. Update max slider
        self.rangeCropVelfieldMaxSlider.setValue(int(value / 5))
        
        #2. If below the set minimum, decrease minimum too
        if value < self.rangeCropVelfieldMin.value():
            # Set the spinbox value
            self.rangeCropVelfieldMax.setValue(value)
            # Set the slider value
            self.rangeCropVelfieldMaxSlider.setValue(int(value / 5))
    
    def actionRunSetup(self):        
        # Start the progress bar
        self.startProgressBar('Running algorithm setup')
        
        # Determine the method number
        if self.mainSettingsMethodNumberGroupBoxButtonJux.isChecked():
            self.methodNumber = '0'
            self.methodName = "Jux' linear extraction"
            coinInfo = {}
        elif self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp.isChecked():
            self.methodNumber = '1'
            self.methodName = "Bin-constrained linear interpolation"
            coinInfo = {}
        elif self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit.isChecked():
            self.methodNumber = '2'
            self.methodName = "Bin-nudged quadratic fit"
            coinInfo = {}
        elif self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit.isChecked():
            self.methodNumber = '3'
            self.methodName = "Tracer-nudged quadratic fit"
            coinInfo = {}
        elif self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit.isChecked():
            self.methodNumber = '4'
            self.methodName = "Tracer-constrained quadratic fit"
            coinInfo = {}
        elif self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth.isChecked():
            self.methodNumber = 'GT'
            self.methodName = "Tracer coin-based wall-normal fit"
            
            dictMapCoinFit = {0: 'LIN',
                              1: 'QUAD',
                              2: 'CUBIC',
                              3: 'CLAUSER',
                              -1: 'NONE'
                              }
            
            # Determine coinFit method
            coinInfo = {'coinFitConstrained': self.mainSettingsCoinFitCheckBoxConstrain.isChecked(),
                        'coinHeight': self.HCoin,
                        'coinOverlap': self.mainSettingsCoinOverlapValue.value(),
                        'coinFitMethod': dictMapCoinFit[self.mainSettingsCoinFitButtonGroup.checkedId()]}
            
            # useObjectInfo = coinInfo['coinFitConstrained']
            # useLSRConstrained = False
            # VOItype='c'
            # coinHeight = coinInfo['coinHeight']
            # coinOverlap = coinInfo['coinOverlap']
            # fitMethod = coinInfo['coinFitMethod']
        else:
            logging.erro('No method selected!!')
            return
        
        if hasattr(self, 'pitchMM'):
            grid_size = self.pitchMM
        else:
            grid_size = None
        
        logging.info(f'Selected method {self.methodNumber}: {self.methodName}')
        logging.info(f'Radius of interrogation sphere: {self.RSphere}')
        
        # Clean up any old threads
        if hasattr(self, 'threadMainSolver'):
            del self.threadMainSolver
        
        # Extract sphere slicing settings
        self.useSlicer = self.mainSettingsSliceOnCheckbox.isChecked()
        self.isCutByPlane = self.mainSettingsSliceSphereRadiusUsePlane.isChecked()
        self.RSphereLargeToCutWith = self.mainSettingsSliceSphereRadiusSpinBox.value()
        
        useGroundPlaneFit = hasattr(self, 'tracerDataGF')
        if useGroundPlaneFit:
            tracerDataGFToAdd = self.tracerDataGF.iloc[:,:3]
        else:
            tracerDataGFToAdd = None
        
        # Create the worker for the main solver (is to be done once and then used in the other approaches)
        self.threadMainSolver = MainSolverWorker(self.RSphere, self.methodNumber, coinInfo=coinInfo, useGroundPlaneFit=useGroundPlaneFit)
        self.threadMainSolver.progress.connect(self.updateProgressBar)
        self.threadMainSolver.logSignal.connect(self.logThreaded)
        self.threadMainSolver.finishedSetup.connect(self.setupFinished)
        
        
        
        # Execute setup
        self.threadMainSolver.executeSetup(self.vtkOutputMesh, self.fluidDataVelfield,
                                           (self.rangeCropVelfieldMin.value(),
                                            self.rangeCropVelfieldMax.value()),
                                           self.fluidDataTypeIsBin,
                                           modelPD = self.modelSTLreader.GetOutput(),
                                           modelPDFilePath = self.modelPDFileName,
                                           coorTracers = tracerDataGFToAdd,
                                           coorTracersCropDim = (self.rangeCropTracersMin.value(),
                                                                 self.rangeCropTracersMax.value()),
                                           useSlice = self.useSlicer,
                                           usePlaneSlice = self.isCutByPlane,
                                           rSliceSphere = self.RSphereLargeToCutWith,
                                           grid_size = grid_size
                                           )
    @QtCore.pyqtSlot()
    def setupFinished(self):
        # Reset the progress bar
        self.resetProgressBar()
        
        # Enable the evaluation actions
        self.evaluationGroupBox.setEnabled(True)
        
        self.setupIsFinished = True
        
        # Log results
        logging.info(f'Done with setup for method {self.methodNumber}: {self.methodName}')
        
    def resetPlots(self, loaded=False):
        # self.mainPlot.clear()
        # self.mainPlot.subplot(0)
        self.mainPlot.subplot(0, 3)
        if not loaded:
            self.mainPlotSubplotsList[0] = pv.Chart2D()
        self.mainPlot.add_chart(self.mainPlotSubplotsList[0])
        
        # self.mainPlot.subplot(1)
        self.mainPlot.subplot(1, 3)
        if not loaded:
            self.mainPlotSubplotsList[1] = pv.Chart2D()
        self.mainPlot.add_chart(self.mainPlotSubplotsList[1])
        # self.mainPlot.add_axes(interactive=True)
        
        # self.mainPlot.subplot(2)
        self.mainPlot.subplot(2, 3)
        if not loaded:
            self.mainPlotSubplotsList[2] = pv.Chart2D()
        self.mainPlot.add_chart(self.mainPlotSubplotsList[2])
        # self.mainPlot.add_axes(interactive=True)
        
        self.mainPlot.subplot(3, 0)
        if not loaded:
            self.mainPlotSubplotsList[3] = pv.Chart2D()
        self.mainPlot.add_chart(self.mainPlotSubplotsList[3])
        
        self.mainPlot.subplot(3, 1)
        if not loaded:
            self.mainPlotSubplotsList[4] = pv.Chart2D()
        self.mainPlot.add_chart(self.mainPlotSubplotsList[4])
        
        self.mainPlot.subplot(3, 2)
        if not loaded:
            self.mainPlotSubplotsList[5] = pv.Chart2D()
        self.mainPlot.add_chart(self.mainPlotSubplotsList[5])
        
        
        # self.mainPlot.subplot(3)
        self.mainPlot.subplot(0,0)
        # self.mainPlotSubplotsList[2] = self.get3DPlot()
        self.mainPlot.add_axes(interactive=True)
        self.mainPlot.enable_terrain_style(mouse_wheel_zooms=True)
        # self.mainPlot.add_axes(interactive=True)
        
        # self.mainPlot.subplot(1,1)
        # self.mainPlotSubplotsList[1,1] = pv.Chart2D()
        # self.mainPlot.add_chart(self.mainPlotSubplotsList[1,1])
        # # self.mainPlot.add_axes(interactive=True)
        
        # self.mainPlot.subplot(1,2)
        # self.mainPlotSubplotsList[1,2] = pv.Chart2D()
        # self.mainPlot.add_chart(self.mainPlotSubplotsList[1,2])
        # # self.mainPlot.add_axes(interactive=True)
        
    def actionTakePicture(self):
        # Create a window to image filter
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        
        
        # Check what is the active view
        active_view_tab = self.viewerTabsWidget.currentIndex()
        if active_view_tab == 0:
            render_window = self.vtkWidgetViewer.GetRenderWindow()
            filename_extra = '3DViewer-'
        else:
            render_window = self.mainPlot.render_window
            filename_extra = 'PlotViewer-'
        
        # Add the render window
        windowToImageFilter.SetInput(render_window)
        
        if (vtk.VTK_MAJOR_VERSION >= 8) or ((vtk.VTK_MAJOR_VERSION == 8) and (vtk.VTK_MINOR_VERSION >= 90)):
            windowToImageFilter.SetScale(2)
        else:
            windowToImageFilter.SetMagnification(2)
        
        # Read the opacity (alpha-value) as well
        windowToImageFilter.SetInputBufferTypeToRGBA()
        
        # Always read from the front buffer (so what you see is what you get)
        windowToImageFilter.ReadFrontBufferOff()
        
        # Update filter
        windowToImageFilter.Update()
        
        # Save the image to a file
        PNGwriter = vtk.vtkPNGWriter()
        filepath = self.screenShotDir + r'\LIVA-processor-' + filename_extra + datetime.datetime.now().strftime('%Y-%m-%dH%H%M%S%f') + '.png'
        PNGwriter.SetFileName(filepath)
        PNGwriter.SetInputData(windowToImageFilter.GetOutput())
        PNGwriter.Write()
        
        # Fix the zoom-in
        self.vtkWidgetViewer.GetRenderWindow().Render()
    
    def actionEvaluateSingleIterate(self):
        # Get the cell id for evaluation
        cell_idx = self.singleEvaluationIdx.value()
        
        # Start the progress bar
        self.startProgressBar(f'Running single evaluation for cell {self.singleEvaluationIdx.value()}')
        
        # Set up actions for after
        self.threadMainSolver.CSSA.ShowEvaluationAtCellById(cell_idx,
                                                            plotResults=False
                                                            )
        
        # Clear old plotted data
        if self.chartDataLoaded:
            self.mainPlot.clear_actors()
            self.resetPlots(self.chartDataLoaded)
            for item in self.mainPlotSubplotsList:
                if not isinstance(item, type(None)):
                    item.clear()
        
        obj = self.threadMainSolver.CSSA
        velfield_rot = obj.funcRotateBackward(obj.velfield_in_sphere[:,:3])
        ######################
        # Plot the results
        ######################
        ### 3D Plot
        # Add a sphere around the point
        if self.chartDataLoaded:
            if (self.methodNumber == '0') or (self.methodNumber == 'GT'):
                # Construct the transformation matrix (the cylinder initially points in +y direction)
                sourceVector = np.array([0., 1., 0.])
                rotationMatrix = pivDA.GetRodriguesTransformationMatrix(sourceVector, obj.normal)
                vtkRotation = vtk.vtkTransform()
                vtkRotation.SetMatrix(rotationMatrix.flatten())
                vtkRotation.Update()
                
                # Define the new cylinder center
                if self.methodNumber == '0':
                    maxH = obj.maxH
                else:
                    maxH = obj.cylinderHeight
                if self.methodNumber == '0':
                    cylinderCenter = obj.point + maxH/2 * obj.normal
                else:
                    cylinderCenter = obj.point + (maxH-obj.coinHeight)/2 * obj.normal
                
                # # Transform the cylinder to align with the cell normal
                self.baseCylinderTransform = vtk.vtkTransform()
                self.baseCylinderTransform.PostMultiply()
                self.baseCylinderTransform.Translate(*cylinderCenter)
                self.baseCylinderTransform.SetInput(vtkRotation)
                
                self.interrogationVolume.SetTransform(self.baseCylinderTransform)
            else:
                if self.useSlicer:
                    if self.isCutByPlane:
                        # Update sphere position
                        self.sphere_small.SetCenter(obj.point)
                        
                        # Update plane position and normal
                        self.cutPlane.SetNormal(obj.normal)
                        self.cutPlane.SetOrigin(obj.point)
                    else:
                        # Determine the rotation
                        sourceVector = np.array([0., 0., 1.])
                        rotationMatrix = pivDA.GetRodriguesTransformationMatrix(sourceVector, obj.normal)
                        vtkRotation = vtk.vtkTransform()
                        vtkRotation.SetMatrix(rotationMatrix.flatten())
                        vtkRotation.Update()
                        
                        # Update transformation to polydata filter
                        self.baseSlicedSphereTransform = vtk.vtkTransform()
                        self.baseSlicedSphereTransform.PostMultiply()
                        self.baseSlicedSphereTransform.Translate(*obj.point)
                        self.baseSlicedSphereTransform.SetInput(vtkRotation)
                        
                        
                        self.interrogationVolume.SetTransform(self.baseSlicedSphereTransform)
                else:
                    # Update the sphere iterate center
                    self.interrogationVolume.SetCenter(obj.point)
                    
            # Finally call Update to the generic interrogationVolume to update interrogation volume
            self.interrogationVolume.Update()
        else:
            if (self.methodNumber == '0') or (self.methodNumber == 'GT'):
                # Construct the transformation matrix (the cylinder initially points in +y direction)
                sourceVector = np.array([0., 1., 0.])
                rotationMatrix = pivDA.GetRodriguesTransformationMatrix(sourceVector, obj.normal)
                vtkRotation = vtk.vtkTransform()
                vtkRotation.SetMatrix(rotationMatrix.flatten())
                vtkRotation.Update()
                
                # Define the cylinder center
                if self.methodNumber == '0':
                    maxH = obj.maxH
                else:
                    maxH = obj.cylinderHeight
                if self.methodNumber == '0':
                    cylinderCenter = obj.point + maxH/2 * obj.normal
                else:
                    cylinderCenter = obj.point + (maxH-obj.coinHeight)/2 * obj.normal
                
                # Create the cylinder
                baseCylinder = vtk.vtkCylinderSource()
                baseCylinder.SetCenter((0, 0, 0))
                baseCylinder.SetRadius(obj.R)
                baseCylinder.SetHeight(maxH)
                baseCylinder.SetResolution(100)
                baseCylinder.Update()
                
                # Transform the cylinder to align with the cell normal
                self.baseCylinderTransform = vtk.vtkTransform()
                self.baseCylinderTransform.PostMultiply()
                self.baseCylinderTransform.Translate(*cylinderCenter)
                self.baseCylinderTransform.SetInput(vtkRotation)
                self.baseCylinderTransform.Update()
                
                self.interrogationVolume = vtk.vtkTransformPolyDataFilter()
                self.interrogationVolume.SetTransform(self.baseCylinderTransform)
                self.interrogationVolume.SetInputConnection(baseCylinder.GetOutputPort())
                self.interrogationVolume.Update()
            else:
                # Add a sphere that is cut
                if self.useSlicer:
                    if self.isCutByPlane:
                        self.sphere_small = vtk.vtkSphereSource()
                        self.sphere_small.SetRadius(self.RSphere)
                        self.sphere_small.SetThetaResolution(20)
                        self.sphere_small.SetPhiResolution(20)
                        self.sphere_small.SetCenter(obj.point)
                        self.sphere_small.Update()
                    
                        self.cutPlane = vtk.vtkPlane()
                        self.cutPlane.SetNormal(obj.normal)
                        self.cutPlane.SetOrigin(obj.point)
                        
                        self.interrogationVolume = vtk.vtkPolyDataPlaneClipper()
                        self.interrogationVolume.SetPlane(self.cutPlane)
                        self.interrogationVolume.SetInputConnection(self.sphere_small.GetOutputPort())
                        self.interrogationVolume.Update()
                    else:
                        sourceVector = np.array([0., 0., 1.])
                        rotationMatrix = pivDA.GetRodriguesTransformationMatrix(sourceVector, obj.normal)
                        vtkRotation = vtk.vtkTransform()
                        vtkRotation.SetMatrix(rotationMatrix.flatten())
                        vtkRotation.Update()
                        
                        # Create the slicer sphere
                        r_large = self.RSphereLargeToCutWith
                        sphere_slicer = vtk.vtkSphereSource()
                        sphere_slicer.SetRadius(r_large)
                        sphere_slicer.SetThetaResolution(20)
                        sphere_slicer.SetPhiResolution(20)
                        sphere_slicer.SetCenter(0., 0., -r_large)
                        sphere_slicer.Update()
                        
                        # Create the main sphere
                        sphere_main = vtk.vtkSphereSource()
                        sphere_main.SetRadius(self.RSphere)
                        sphere_main.SetThetaResolution(20)
                        sphere_main.SetPhiResolution(20)
                        sphere_main.SetCenter(0., 0., 0., )
                        sphere_main.Update()
                        
                        # Create the sliced sphere
                        slicedSphere = vtk.vtkBooleanOperationPolyDataFilter()
                        slicedSphere.SetInputConnection(0, sphere_main.GetOutputPort())
                        slicedSphere.SetInputConnection(1, sphere_slicer.GetOutputPort())
                        slicedSphere.SetOperationToDifference()
                        slicedSphere.Update()
                        
                        # Transform the sliced sphere to align with the cell normal
                        self.baseSlicedSphereTransform = vtk.vtkTransform()
                        self.baseSlicedSphereTransform.PostMultiply()
                        self.baseSlicedSphereTransform.Translate(*obj.point)
                        self.baseSlicedSphereTransform.SetInput(vtkRotation)
                        self.baseSlicedSphereTransform.Update()
                        
                        # Transform the sliced sphere (rotation + translation)
                        self.interrogationVolume = vtk.vtkTransformPolyDataFilter()
                        self.interrogationVolume.SetTransform(self.baseSlicedSphereTransform)
                        self.interrogationVolume.SetInputConnection(slicedSphere.GetOutputPort())
                        self.interrogationVolume.Update()
                else:
                    # Add the sphere
                    self.interrogationVolume = vtk.vtkSphereSource()
                    self.interrogationVolume.SetRadius(self.RSphere)
                    self.interrogationVolume.SetThetaResolution(20)
                    self.interrogationVolume.SetPhiResolution(20)
                    self.interrogationVolume.SetCenter(obj.point)
                    self.interrogationVolume.Update()
            
            # Create a mapper 
            self.interrogationVolumeMapper = vtk.vtkPolyDataMapper()
            self.interrogationVolumeMapper.SetInputConnection(self.interrogationVolume.GetOutputPort())
            
            # Create an actor
            self.interrogationVolumeActor = vtk.vtkActor()
            self.interrogationVolumeActor.SetMapper(self.interrogationVolumeMapper)
            self.interrogationVolumeActor.GetProperty().SetRepresentationToWireframe()
            self.interrogationVolumeActor.GetProperty().SetColor(self.vtkColors.GetColor3d("Banana"))
            self.interrogationVolumeActor.GetProperty().SetLineWidth(3)
            
            # Add actor to the widget renderer
            self.vtkWidgetRen.AddActor(self.interrogationVolumeActor)
            
        # Update renderer
        self.vtkWidgetViewer.GetRenderWindow().Render()
        self.vtkWidgetIren.Initialize()
        
        ##################
        ### 2D Plot
        # Define pre-variables
        xmin = obj.point[0] - obj.R
        xmax = obj.point[0] + obj.R
        
        ymin = obj.point[1] - obj.R
        ymax = obj.point[1] + obj.R
        
        zmin = obj.point[2] - obj.R
        zmax = obj.point[2] + obj.R
        
        # Set the scope to the first subplot (side view, xz-plane)
        chart = self.mainPlotSubplotsList[0]
        chart.scatter(velfield_rot[obj.NFMPoints:,0] + obj.point[0], velfield_rot[obj.NFMPoints:,2] + obj.point[2],
                      color='tab:red', style='o', label='Object points')
        chart.scatter(velfield_rot[:obj.NFMPoints,0] + obj.point[0], velfield_rot[:obj.NFMPoints,2] + obj.point[2],
                      color='tab:blue', style='o', label='Fluid points')
        chart.x_label = 'X [mm]'
        chart.y_label = 'Z [mm]'
        chart.x_axis.range = [xmin, xmax]
        chart.y_axis.range = [zmin, zmax]
        chart.title = 'Side view in x-z plane'
        
        # Set the scope to the second subplot (front view yz-plane)
        chart = self.mainPlotSubplotsList[1]
        chart.scatter(velfield_rot[obj.NFMPoints:,1] + obj.point[1], velfield_rot[obj.NFMPoints:,2] + obj.point[2],
                      color='tab:red', style='o', label='Object points')
        chart.scatter(velfield_rot[:obj.NFMPoints,1] + obj.point[1], velfield_rot[:obj.NFMPoints,2] + obj.point[2],
                      color='tab:blue', style='o', label='Fluid points')
        chart.x_label = 'Y [mm]'
        chart.y_label = 'Z [mm]'
        chart.title = 'Front view in y-z plane'
        chart.x_axis.range = [ymax, ymin]
        chart.y_axis.range = [zmin, zmax]
        
        # Set the scope to the second subplot (top view, xy-plane)
        chart = self.mainPlotSubplotsList[2]
        chart.scatter(velfield_rot[obj.NFMPoints:,0] + obj.point[0], velfield_rot[obj.NFMPoints:,1] + obj.point[1],
                      color='tab:red', style='o', label='Object points')
        chart.scatter(velfield_rot[:obj.NFMPoints,0] + obj.point[0], velfield_rot[:obj.NFMPoints,1] + obj.point[1],
                      color='tab:blue', style='o', label='Fluid points')
        chart.x_label = 'X [mm]'
        chart.y_label = 'Y [mm]'
        chart.title = 'Top view in x-y plane'
        chart.x_axis.range = [xmin, xmax]
        chart.y_axis.range = [ymin, ymax]
        
        
        #### Add solution estimate
        # Define zrange
        if obj.velMethod == '2' or obj.velMethod == '3' or obj.velMethod == '4':
            nrange = np.linspace(min(obj.velfield_in_sphere[:,2].min(), 0), obj.velfield_in_sphere[:,2].max(), 100)
        elif obj.velMethod == 'GT':
            nrange = np.linspace(min(obj.velfield_in_sphere[:,2].min(), 0), obj.cylinderHeight, 100)
        else:
            nrange = np.linspace(min(obj.velfield_in_sphere[:,2].min(), 0), obj.velfield_in_sphere[:,2].max(), 100)
        t1range = np.zeros_like(nrange)
        t2range = np.zeros_like(nrange)
        points = np.c_[t1range, t2range, nrange]
        
        vel_estimate = obj.get_velocity(points)
        umin, umax = (min(vel_estimate[:,0].min(), obj.velfield_in_sphere[:,3].min()),
                      max(vel_estimate[:,0].max(), obj.velfield_in_sphere[:,3].max())
                      )
        vmin, vmax = (min(vel_estimate[:,1].min(), obj.velfield_in_sphere[:,4].min()),
                      max(vel_estimate[:,1].max(), obj.velfield_in_sphere[:,4].max())
                      )
        wmin, wmax = (min(vel_estimate[:,2].min(), obj.velfield_in_sphere[:,5].min()),
                      max(vel_estimate[:,2].max(), obj.velfield_in_sphere[:,5].max())
                      )
        nmin, nmax = (nrange.min(), nrange.max())
        
        # Set the scope to the third subplot with velocity fit along normal
        chart = self.mainPlotSubplotsList[3]
        chart.scatter(obj.velfield_in_sphere[:obj.NFMPoints,3], obj.velfield_in_sphere[:obj.NFMPoints,2], color='black',  style='o', label='Data')
        chart.line(vel_estimate[:,0], nrange, color='tab:red', label='Velocity fit')
        chart.x_label = 'u [m/s]'
        chart.y_label = 'n [mm]'
        chart.title = 'Fit of u-velocity'
        chart.x_axis.range = [umin, umax]
        chart.y_axis.range = [nmin, nmax]
        
        
        # Set the scope to the third subplot with velocity fit along normal
        chart = self.mainPlotSubplotsList[4]
        chart.scatter(obj.velfield_in_sphere[:obj.NFMPoints,4], obj.velfield_in_sphere[:obj.NFMPoints,2], color='black',  style='o', label='Data')
        chart.line(vel_estimate[:,1], nrange, color='tab:red', label='Velocity fit')
        chart.x_label = 'v [m/s]'
        chart.y_label = 'n [mm]'
        chart.title = 'Fit of v-velocity'
        chart.x_axis.range = [vmin, vmax]
        chart.y_axis.range = [nmin, nmax]
        
        # Set the scope to the third subplot with velocity fit along normal
        chart = self.mainPlotSubplotsList[5]
        chart.scatter(obj.velfield_in_sphere[:obj.NFMPoints,5], obj.velfield_in_sphere[:obj.NFMPoints,2], color='black',  style='o', label='Data')
        chart.line(vel_estimate[:,2], nrange, color='tab:red', label='Velocity fit')
        chart.x_label = 'w [m/s]'
        chart.y_label = 'n [mm]'
        chart.title = 'Fit of w-velocity'
        chart.x_axis.range = [wmin, wmax]
        chart.y_axis.range = [nmin, nmax]
        
        ###################
        ## 3D plot of sphere with fluid data and intersected geometry
        # self.mainPlot.subplot(3)
        self.mainPlot.subplot(0,0)
        
        # Add the model STL but opaque
        modelSTL_pyvista = pv.wrap(self.modelSTLreader.GetOutput())
        self.mainPlot.add_mesh(modelSTL_pyvista, opacity=0.3)
        
        # Add the model plane if added
        if self.groundPlaneExists:
            gp_pyvista = pv.wrap(self.modelPlane.GetOutput())
            self.mainPlot.add_mesh(gp_pyvista, opacity=0.3)
            
        # Add the sphere
        spherePV = pv.wrap(self.interrogationVolume.GetOutput())
        self.mainPlot.add_mesh(spherePV, style='wireframe', line_width=3, color=[255,225,53])
        
        # Add the intersection of the object mesh
        if obj.objectIntersect:
            objectMesh = pv.wrap(obj.objectIntersect)
            self.mainPlot.add_mesh(objectMesh, style='surface', show_edges=True, color='Crimson')
        
        # Add the intersection of the fluid mesh
        fluidMesh = pv.PolyData(obj.velfield_in_sphereTrue[:obj.NFMPoints, :])
        self.mainPlot.add_mesh(fluidMesh, style='points', color='Cyan', render_points_as_spheres=True, point_size=8)
        
        # Add the WG poitnts
        if obj.objectIntersect:
            WGMesh = pv.PolyData(obj.velfield_in_sphereTrue[obj.NFMPoints:, :])
            self.mainPlot.add_mesh(WGMesh, style='points', color='#D2B48C', render_points_as_spheres=True, point_size=8)

        self.mainPlot.show_bounds(axes_ranges=[-180, 180, -180, 180, 0, 250])
        self.mainPlot.x_label = 'X [mm]'
        self.mainPlot.y_label = 'Y [mm]'
        self.mainPlot.z_label = 'Z [mm]'
        self.mainPlot.title = '3D View of intersection'
        
        # Set flag
        self.chartDataLoaded = True
        
        logging.info(f'Plotted results for cell {cell_idx}')
        
        # Stop the progress bar
        self.resetProgressBar()
        
    def actionStopMainRun(self):
        # Stop the main algorithm (in some way)
        self.threadMainSolver.terminate()
        
        # Reset the progress bar
        self.resetProgressBar()
        
        # Log info
        logging.error('Main LIVA-algorithm was stopped by user before it finished.')
        
    def actionRunMain(self):
        # Start the progress bar
        self.startProgressBar('Running main LIVA-algorithm')
        
        # Set up actions after ground plane fit is done
        self.threadMainSolver.finishedMainRun.connect(self.mainRunFinished)
        
        # Execute ground plane fit in background
        self.threadMainSolver.executeMainRun()
    
    def mainRunFinished(self, result):
        # Append the results
        pivDA.append_skinFriction_data(self.vtkAdaptedOutputMesh, result, identifier=f'method_{self.methodNumber}')
        
        # Save the results
        methodNameForSaving = self.methodName.replace(' ', '_').replace("'", "")
        savefilename = Path(self.outputMeshFileName).name.replace('.vtk', f'-{methodNameForSaving}.vtu')
        savefilepath = (Path(self.saveDirectory) / savefilename).as_posix()
        
        pivDA.save_adapted_vtk_dataset_as_unstructured_grid(self.vtkAdaptedOutputMesh, savefilepath)
        logging.info(f'Saved method-0 as: {savefilepath}')        
        
        # Log results
        logging.info('Done with main run, the results are saved.')
        
        # Reset the progress bar
        self.resetProgressBar()
        pass
        
    def actionFitGroundPlane(self):
        # Start the progress bar
        self.startProgressBar('Running ground plane fit')
        
        # Set up actions after ground plane fit is done
        self.threadMainSolver.finishedGroundPlaneFit.connect(self.groundPlaneFitFinished)
        
        # Execute ground plane fit in background
        self.threadMainSolver.executeGroundPlaneFit(self.groundPlaneFitFolder)
        
    # @QtCore.pyqtSlot(object)
    def groundPlaneFitFinished(self, correctedGPFit):
        # Reset the progress bar
        self.resetProgressBar()
        
        # Save the corrected GP fit: (out = (planeFitNormal, groundPlaneFit.point, d))
        self.GPFit = correctedGPFit
        self.GPFitLoaded = True
        
        # Show ground plane if desired
        self.showGroundPlaneCorrected(self.showGroundPlaneCorrectedCheckButton.isChecked())
        
        # Log results
        logging.info('Done with ground plane fit.')        
    
    def showGroundPlaneCorrected(self, showGroundPlane):
        # Check if it needs to be plotted
        if showGroundPlane and self.GPFitLoaded:
            # Define a model plane
            self.modelPlaneCorrected = vtk.vtkPlaneSource()
            a, b, c = self.GPFit[0]

            # Define two points (X1, Y1) and (X2, Y2) which will lie on the plane
            funcZ = lambda x, y: (-self.GPFit[2] - a*x - b*y) / c
            X1, Y1 = (500.0, 0.0)
            Z1 = funcZ(X1, Y1)
            X2, Y2 = (0.0, 500.0)
            Z2 = funcZ(X2, Y2)
            
            # Define the modelPlane geometry
            # self.modelPlaneCorrected.SetOrigin(np.array(self.GPFit[1]))
            self.modelPlaneCorrected.SetPoint1(X1, Y1, Z1)
            self.modelPlaneCorrected.SetPoint2(X2, Y2, Z2)
            self.modelPlaneCorrected.SetCenter(np.array(self.GPFit[1]))
            self.modelPlaneCorrected.Update()
            
            # Create mappers for the corrected plane
            modelPlaneCorrectedMapper = vtk.vtkPolyDataMapper()
            modelPlaneCorrectedMapper.SetInputConnection(self.modelPlaneCorrected.GetOutputPort())
            
            # Set the model plane representation to a wireframe
            self.modelPlaneActor.GetProperty().SetRepresentationToWireframe()
            self.modelPlaneActor.GetProperty().SetLineWidth(3)
            
            # Create an actor
            self.modelPlaneCorrectedActor = vtk.vtkActor()
            self.modelPlaneCorrectedActor.SetMapper(modelPlaneCorrectedMapper)
            self.modelPlaneCorrectedActor.GetProperty().SetColor(self.vtkColors.GetColor3d("MediumAquaMarine"))
            self.modelPlaneCorrectedActor.GetProperty().SetOpacity(0.7)
            
            # Add actor to renderer
            self.vtkWidgetRen.AddActor(self.modelPlaneCorrectedActor)
            
            # Update vtkWidgetRen
            self.vtkWidgetRen.ResetCamera()
            self.vtkWidgetViewer.GetRenderWindow().Render()
            self.vtkWidgetIren.Initialize()
            
            logging.info('Added corrected ground plane fit to the 3D view')
        else:
            if hasattr(self, 'modelPlaneCorrectedActor'):
                self.vtkWidgetRen.RemoveActor(self.modelPlaneCorrectedActor)
                self.modelPlaneActor.GetProperty().SetRepresentationToSurface()
                logging.info('Corrected ground plane fit removed from 3D view')
        
    def actionUpdateRangeCropTracersDisplay(self):
        txt = f'{str(self.rangeCropTracersMin.value()).ljust(4)} --- {str(self.rangeCropTracersMax.value()).rjust(4)}'
        self.rangeCropTracersDisplay.setText(txt)
        
    def actionUpdateRangeCropVelfieldDisplay(self):
        txt = f'{str(self.rangeCropVelfieldMin.value()).ljust(4)} --- {str(self.rangeCropVelfieldMax.value()).rjust(4)}'
        self.rangeCropVelfieldDisplay.setText(txt)
        
    def InitialiseVTKObjects(self):
        self.vtkColors = vtk.vtkNamedColors()
        return
    
    def CreateVTKWidgetViewer(self):
        # Create a frame to hold the vtk widget
        self.frameVTK = QtWidgets.QFrame()
        
        # Define the layout
        self.layoutFrameVTK = QtWidgets.QVBoxLayout()
        self.vtkWidgetViewer = QVTKRenderWindowInteractor(self.frameVTK)
        self.layoutFrameVTK.addWidget(self.vtkWidgetViewer)
 
        self.vtkWidgetRen = vtk.vtkRenderer()
        self.vtkWidgetViewer.GetRenderWindow().AddRenderer(self.vtkWidgetRen)
        self.vtkWidgetIren = self.vtkWidgetViewer.GetRenderWindow().GetInteractor()
        # Set background color
        self.vtkWidgetRen.SetBackground(self.vtkColors.GetColor3d("Beige"))
        
        # Add axes
        self.AddAxes()
        
        # Change interaction mode to "TrackBall"
        self.vtkWidgetRenStyle = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidgetIren.SetInteractorStyle(self.vtkWidgetRenStyle)
        
        # Render results
        self.vtkWidgetRen.ResetCamera()
        self.vtkWidgetViewer.GetRenderWindow().Render()
        self.vtkWidgetIren.Start()
        
        # Set flags
        self.fluidDataLoaded = False
        self.modelPDLoaded = False
        self.outputMeshLoaded = False
        self.tracerDataLoaded = False
        self.GPFitLoaded = False
        self.singleCellPreviouslyLoaded = False
        self.setupIsFinished = False
        
        return
        
    def CreateLogViewer(self):
        # Create a widget to hold the logging thingies
        self.loggingViewWidget = QtWidgets.QWidget()
        # self.loggingViewWidget.setMaximumWidth(300)
        
        # Create the layout
        self.loggingViewLayout = QtWidgets.QGridLayout()
        
        ##############
        # Add a logging viewer
        self.logTextBox = QtTextEditLogger(self)
        self.logTextBox.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s',
                                                       datefmt='%Y-%m-%d %H:%M:%S'))
        # self.logTextBox.setMinimumWidth(400)
        logging.getLogger().addHandler(self.logTextBox)
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.INFO)
        
        #############
        # Add a view of the main info
        # Create a label
        self.logValuesListTitle = QtWidgets.QLabel('Information:')
        self.logValuesListTitle.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                              QtWidgets.QSizePolicy.Maximum)
        
        # Create a list
        self.logValuesList = QtWidgets.QListWidget()
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        font.setPointSize(6)
        self.logValuesList.setFont(font)
        self.logValuesList.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                         QtWidgets.QSizePolicy.Preferred)
        self.logValuesList.setResizeMode(QtWidgets.QListView.Adjust)
        self.logValuesList.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.logValuesList.viewport().setAutoFillBackground(False)
        self.logValuesList.setVerticalScrollMode(True)
        self.logValuesList.setEnabled(True)
                
        ################
        ### STL data items
        self.logValuesListHeaderObject = QtWidgets.QListWidgetItem()
        self.logValuesListHeaderObject.setText('OBJECT-MESH INFORMATION')
        self.logValuesListSplitterObject = QtWidgets.QListWidgetItem()
        self.logValuesListSplitterObject.setText('--------')
        # Create an item for number of STL points
        self.logValuesListItemObjectNPoints = QtWidgets.QListWidgetItem()
        self.NpObjectString = ''
        # Create an item for number of STL cells
        self.logValuesListItemObjectNCells = QtWidgets.QListWidgetItem()
        self.NcObjectString = ''
        # Create an item for ground plane loaded flag
        self.logValuesListItemObjectGroundPlane = QtWidgets.QListWidgetItem()
        self.GPString = ''
        
        self.updateObjectMeshInfoItems()
        
        
        ################
        ### OutputMesh data items
        self.logValuesListHeaderOM = QtWidgets.QListWidgetItem()
        self.logValuesListHeaderOM.setText('OUTPUT-MESH INFORMATION')
        self.logValuesListSplitterOM = QtWidgets.QListWidgetItem()
        self.logValuesListSplitterOM.setText('--------')
        # Create an item for number of OM-points
        self.logValuesListItemOMNpoints = QtWidgets.QListWidgetItem()
        self.NpOutString = ''
        # Create an item for number of OM-cells
        self.logValuesListItemOMNcells = QtWidgets.QListWidgetItem()
        self.NcOutString = ''
        
        self.updateOutputMeshInfoItems()
        ################
        ### Fluid data items
        self.logValuesListHeaderFM = QtWidgets.QListWidgetItem()
        self.logValuesListHeaderFM.setText('FLUID-MESH INFORMATION')
        self.logValuesListSplitterFM = QtWidgets.QListWidgetItem()
        self.logValuesListSplitterFM.setText('--------')
        # Create an item for pixel to mm distance value
        self.logValuesListItemFMTitle = QtWidgets.QListWidgetItem()
        self.FMTitleString = ''
        # Create an item for bin size value
        self.logValuesListItemBinSize = QtWidgets.QListWidgetItem()
        self.binSizeString = ''
        # Create an item for percentage overlap value
        self.logValuesListItemOverlap = QtWidgets.QListWidgetItem()
        self.overlapString = ''
        # Create an item for pixel to mm distance value
        self.logValuesListItemPixelToMM_PXpMM = QtWidgets.QListWidgetItem()
        self.PXpMMString = ''
        # Create an item for pixel to mm distance value
        self.logValuesListItemPitchPX = QtWidgets.QListWidgetItem()
        self.pitchPXString = ''
        # Create an item for pixel to mm distance value
        self.logValuesListItemPitchMM = QtWidgets.QListWidgetItem()
        self.pitchMMString = ''
        # Create an item for number of fluid data points
        self.logValuesListItemNumberOfFMPoints = QtWidgets.QListWidgetItem()
        self.NpFMString = ''
        
        self.updateFluidMeshInfoItems()
        ################
        ### Tracer data items
        self.logValuesListHeaderTD = QtWidgets.QListWidgetItem()
        self.logValuesListHeaderTD.setText('TRACER-DATA INFORMATION')
        # Create an item for number of tracer data points
        self.logValuesListItemNumberOfTDPoints = QtWidgets.QListWidgetItem()
        self.NpTDString = ''
        
        self.updateTracerDataInfoItems()
        
        ############################
        # Add all items to the list
        self.logValuesList.addItem(self.logValuesListHeaderObject)
        self.logValuesList.addItem(self.logValuesListItemObjectNPoints)
        self.logValuesList.addItem(self.logValuesListItemObjectNCells)
        self.logValuesList.addItem(self.logValuesListItemObjectGroundPlane)
        self.logValuesList.addItem(self.logValuesListSplitterObject)
        
        self.logValuesList.addItem(self.logValuesListHeaderOM)
        self.logValuesList.addItem(self.logValuesListItemOMNpoints)
        self.logValuesList.addItem(self.logValuesListItemOMNcells)
        self.logValuesList.addItem(self.logValuesListSplitterOM)
        
        self.logValuesList.addItem(self.logValuesListHeaderFM)
        self.logValuesList.addItem(self.logValuesListItemFMTitle)
        self.logValuesList.addItem(self.logValuesListItemBinSize)
        self.logValuesList.addItem(self.logValuesListItemOverlap)
        self.logValuesList.addItem(self.logValuesListItemPitchPX)
        self.logValuesList.addItem(self.logValuesListItemPitchMM)
        self.logValuesList.addItem(self.logValuesListItemPixelToMM_PXpMM)
        self.logValuesList.addItem(self.logValuesListItemNumberOfFMPoints)
        self.logValuesList.addItem(self.logValuesListSplitterFM)
        
        self.logValuesList.addItem(self.logValuesListHeaderTD)
        self.logValuesList.addItem(self.logValuesListItemNumberOfTDPoints)
        
        
        self.logValuesList.setMinimumHeight(self.logValuesList.sizeHintForRow(0) * self.logValuesList.count())
        self.logValuesList.setMinimumWidth(350)
        
        ##############
        # Add a loading display inside a groupbox
        self.loadDisplayGroupBox = QtWidgets.QGroupBox("Track Progress")
        
        # Create layout for the widgets
        self.loadDisplayGroupBoxLayout = QtWidgets.QGridLayout()
        
        ## Create the widgets
        # Progress bar
        self.loadDisplayProgress = QtWidgets.QProgressBar()
        self.loadDisplayProgress.setValue(0)
        # Display text
        self.loadDisplayText = QtWidgets.QLabel()
        self.loadDisplayText.setAlignment(QtCore.Qt.AlignCenter)
        self.loadDisplayText.setText(self._baseProgressText)
        # The spinner
        self.loadDisplaySpinner = QtWaitingSpinner(centerOnParent=False)
        self.loadDisplaySpinner.logSignal.connect(self.logThreaded)
        
        # Add widgets to the layout
        self.loadDisplayGroupBoxLayout.addWidget(self.loadDisplayProgress, 0, 0, 1, 4)
        self.loadDisplayGroupBoxLayout.addWidget(self.loadDisplayText,     1, 0, 1, 3)
        self.loadDisplayGroupBoxLayout.addWidget(self.loadDisplaySpinner,  1, 3, 1, 1)
        
        # Add VBoxLayout to the GroupBox widget
        self.loadDisplayGroupBox.setLayout(self.loadDisplayGroupBoxLayout)
        
        ######
        # Add widgets to the layout
        self.loggingViewLayout.addWidget(self.logTextBox.widget, 0, 0, 4, 1)
        self.loggingViewLayout.addWidget(self.logValuesListTitle, 4, 0, 1, 1)
        self.loggingViewLayout.addWidget(self.logValuesList, 5, 0, 2, 1)
        self.loggingViewLayout.addWidget(self.loadDisplayGroupBox, 7, 0, 1, 1)
        
        # Add layout to the parent widget
        self.loggingViewWidget.setLayout(self.loggingViewLayout)
        
        return
    
    def updateObjectMeshInfoItems(self):
        self.logValuesListItemObjectNPoints.setText('# of STL Points [-] ($NpObject): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.NpObjectString, '...'))
        self.logValuesListItemObjectNCells.setText('# of STL Cells [-] ($NcObject): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.NcObjectString, '...'))
        self.logValuesListItemObjectGroundPlane.setText('Ground plane included [-] ($GP): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.GPString, '...'))
    
    def updateOutputMeshInfoItems(self):
        self.logValuesListItemOMNpoints.setText('# of OutputMesh points [-] ($NpOut): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.NpOutString, '...'))
        self.logValuesListItemOMNcells.setText('# of cells [-] ($NcOut): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.NcOutString, '...'))
        
    def updateFluidMeshInfoItems(self):
        self.logValuesListItemFMTitle.setText('FluidMesh title ($FMTitle): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.FMTitleString, '...'))
        self.logValuesListItemBinSize.setText('Bin size [px]  ($binSize): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.binSizeString, '...'))
        self.logValuesListItemOverlap.setText('Overlap [%] ($overlap): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.overlapString, '...'))
        self.logValuesListItemPixelToMM_PXpMM.setText('Pixel per mm ($PXpMM): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.PXpMMString, '...'))
        self.logValuesListItemPitchPX.setText('Pitch [px] ($pitchPX): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.pitchPXString, '...'))
        self.logValuesListItemPitchMM.setText('Pitch [mm] ($pitchMM): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.pitchMMString, '...'))
        self.logValuesListItemNumberOfFMPoints.setText('# of FluidMesh points [-] ($NpFM): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.NpFMString, '...'))
        
    def updateTracerDataInfoItems(self):
        self.logValuesListItemNumberOfTDPoints.setText('# of TracerData points [-] ($NpTD): '.ljust(self._NUMBER_OF_CHARACTERS_IN_STRING) + ifempty(self.NpTDString, '...'))

    @QtCore.pyqtSlot(str)
    def logThreaded(self, logString):
        logging.info(logString)
        
    def startSpinner(self):
        # Start the thread which controls the spinner
        self.loadDisplaySpinner.start()
    
    def stopSpinner(self):
        # Call the thread to stop the spinning
        self.loadDisplaySpinner.stopSpinning()
    
    def startProgressBar(self, displayText):
        self.loadDisplayText.setText(displayText)
        self.loadDisplayProgress.setValue(0)
        self.startSpinner()
        
    @QtCore.pyqtSlot(int)
    def updateProgressBar(self, value):
        self.loadDisplayProgress.setValue(value)
    
    @QtCore.pyqtSlot()
    def resetProgressBar(self):
        self.loadDisplayText.setText(self._baseProgressText)
        self.loadDisplayProgress.setValue(0)
        self.stopSpinner()
        
    
    def AddAxes(self):
        # Add 3D-axes
        self.axesActor = vtk.vtkAxesActor()
        # Create orientationMarkerWidget
        self.axesWidget = vtk.vtkOrientationMarkerWidget()
        # Set colors of the orientationMarker
        rgba = [0] * 4
        self.vtkColors.GetColor('Carrot', rgba)
        self.axesWidget.SetOutlineColor(*rgba[:3])
        # Add to actor
        self.axesWidget.SetOrientationMarker(self.axesActor)
        self.axesWidget.SetInteractor(self.vtkWidgetIren)
        self.axesWidget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axesWidget.SetEnabled(True)
        self.axesWidget.InteractiveOff()
        return
    
    def CreateVTKWidgetController(self):
        # Create the main controller widget
        self.controllerWidgetGeometry = QtWidgets.QWidget()
        self.controllerWidgetFluid = QtWidgets.QWidget()
        
        # Create Layout to hold all loading controls
        self.controllerWidgetGeometryLayout = QtWidgets.QVBoxLayout()
        self.controllerWidgetFluidLayout = QtWidgets.QVBoxLayout()
        
        # Divide the layout in two
        self.controllerGeometryGroupBox = QtWidgets.QGroupBox('Geometrical data')
        self.controllerGeometryGroupBoxLayout = QtWidgets.QVBoxLayout()
        
        self.controllerFluidGroupBox = QtWidgets.QGroupBox('Fluid-measurement data')
        self.controllerFluidGroupBoxLayout = QtWidgets.QVBoxLayout()
        
        #######################################################################
        #######  Create and add a group box to load object STL polydata
        self.modelPDGroupBox = QtWidgets.QGroupBox("Define Object STL")
        # self.modelPDGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
        #                                    QtWidgets.QSizePolicy.Maximum)
        self.modelPDGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Add display bar
        self.loadModelPDDisplayFile = QtWidgets.QLineEdit()
        self.loadModelPDDisplayFile.readOnly = True
        # Add pushbutton to choose file
        self.chooseModelPDButton = QtWidgets.QPushButton("Choose STL")
        self.chooseModelPDButton.clicked.connect(self.actionOpenModelPDFileDialog)
        self.modelPDFileName = ''
        # Add setting for adding ground plane
        self.groundPlaneCheckBox = QtWidgets.QCheckBox()
        self.groundPlaneCheckBox.setText('Add ground plane')
        self.groundPlaneZPosBox = QtWidgets.QDoubleSpinBox()
        self.groundPlaneZPosBox.setValue(0.)
        self.groundPlaneZPosBox.setEnabled(False)
        self.groundPlaneZPosBox.setSuffix(' mm')
        self.groundPlaneCheckBox.stateChanged.connect(self.groundPlaneZPosBox.setEnabled)
        # Add load button
        self.loadModelPDButton = QtWidgets.QPushButton("Load STL file...")
        self.loadModelPDButton.clicked.connect(self.actionLoadModelPD)
        # Add clear button
        self.clearModelPDButton = QtWidgets.QPushButton("Clear STL from viewer...")
        self.clearModelPDButton.clicked.connect(self.actionClearModelPD)
        
        # CheckButton for showing model STL (polydata)
        self.modelPDShowCheckButton = QtWidgets.QCheckBox('Show Model STL')
        self.modelPDShowCheckButton.setChecked(True)
        self.modelPDShowCheckButton.stateChanged.connect(self.showModelPD)
        
        # Add widgets to layout
        self.modelPDGroupBoxLayout.addWidget(self.loadModelPDDisplayFile, 0, 0, 1, 3)
        self.modelPDGroupBoxLayout.addWidget(self.chooseModelPDButton, 0, 3, 1, 1)
        self.modelPDGroupBoxLayout.addWidget(self.groundPlaneCheckBox, 1, 1, 1, 1)
        self.modelPDGroupBoxLayout.addWidget(self.groundPlaneZPosBox, 1, 2, 1, 1)
        self.modelPDGroupBoxLayout.addWidget(self.modelPDShowCheckButton, 2, 0, 1, 1)
        self.modelPDGroupBoxLayout.addWidget(self.loadModelPDButton, 2, 1, 1, 1)
        self.modelPDGroupBoxLayout.addWidget(self.clearModelPDButton, 2, 2, 1, 2)
        
        # Finally set the layout to the group box
        self.modelPDGroupBox.setLayout(self.modelPDGroupBoxLayout)
        
        #######################################################################
        #######  Group box for output-mesh
        self.outputMeshGroupBox = QtWidgets.QGroupBox("Define OutputMesh")
        # self.outputMeshGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
        #                                       QtWidgets.QSizePolicy.Maximum)
        self.outputMeshGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Add display bar
        self.loadOutputMeshDisplayFile = QtWidgets.QLineEdit()
        self.loadOutputMeshDisplayFile.readOnly = True
        # Add pushbutton to choose file
        self.chooseOutputMeshButton = QtWidgets.QPushButton("Choose OutputMesh VTK file")
        self.chooseOutputMeshButton.clicked.connect(self.actionOpenOutputMeshFileDialog)
        self.outputMeshFileName = ''
        # Add pushbutton to create outputmesh from STL Polydata
        self.createOutputMeshFromSTLButton = QtWidgets.QPushButton("Create Outputmesh from STL")
        self.createOutputMeshFromSTLButton.clicked.connect(self.actionCreateOutputMeshFromSTL)
        self.createOutputMeshFromSTLButton.setEnabled(False)
        # Add load button
        self.loadOutputMeshButton = QtWidgets.QPushButton("Load OutputMesh VTK file...")
        self.loadOutputMeshButton.clicked.connect(self.actionLoadOutputMesh)
        # Add clear button
        self.clearOutputMeshButton = QtWidgets.QPushButton("Clear OutputMesh...")
        self.clearOutputMeshButton.clicked.connect(self.actionClearOutputMesh)
        
        # CheckButton for showing output mesh
        self.outputMeshShowCheckButton = QtWidgets.QCheckBox('Show OutputMesh')
        self.outputMeshShowCheckButton.setChecked(True)
        self.outputMeshShowCheckButton.stateChanged.connect(self.showOutputMesh)
        
        # Add widgets to layout
        self.outputMeshGroupBoxLayout.addWidget(self.loadOutputMeshDisplayFile, 0, 0, 1, 3)
        self.outputMeshGroupBoxLayout.addWidget(self.chooseOutputMeshButton, 0, 3, 1, 1)
        self.outputMeshGroupBoxLayout.addWidget(self.outputMeshShowCheckButton, 1, 0, 1, 1)
        self.outputMeshGroupBoxLayout.addWidget(self.loadOutputMeshButton, 1, 1, 1, 1)
        self.outputMeshGroupBoxLayout.addWidget(self.clearOutputMeshButton, 1, 2, 1, 2)
        self.outputMeshGroupBoxLayout.addWidget(self.createOutputMeshFromSTLButton, 2, 0, 1, -1)
        
        # Finally set the layout to the group box
        self.outputMeshGroupBox.setLayout(self.outputMeshGroupBoxLayout)
        
        self.controllerGeometryGroupBoxLayout.addWidget(self.modelPDGroupBox)
        self.controllerGeometryGroupBoxLayout.addWidget(self.outputMeshGroupBox)
        self.controllerGeometryGroupBox.setLayout(self.controllerGeometryGroupBoxLayout)
        
        #######################################################################
        #######  Group box for velfield data (= fluid-mesh)
        self.fluidDataGroupBox = QtWidgets.QGroupBox("Define FluidData")
        self.fluidDataGroupBoxLayout = QtWidgets.QGridLayout()
        # self.fluidDataGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
        #                                      QtWidgets.QSizePolicy.Preferred)
        
        # Add display bar
        self.loadFluidDataDisplayFile = QtWidgets.QLineEdit()
        self.loadFluidDataDisplayFile.readOnly = True
        # Add pushbutton to choose file
        self.chooseFluidDataButton = QtWidgets.QPushButton("Choose FluidMesh .dat/.npy")
        self.chooseFluidDataButton.clicked.connect(self.actionOpenFluidDataFileDialog)
        self.fluidDataFileName = ''
        # Add load button
        self.loadFluidDataButton = QtWidgets.QPushButton("Load FluidData file...")
        self.loadFluidDataButton.clicked.connect(self.actionLoadFluidDataThreaded)
        # Add button group
        self.fluidDataTypeGroupBox = QtWidgets.QGroupBox("FluidData Type")
        # self.fluidDataTypeGroupBox.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
        #                                          QtWidgets.QSizePolicy.Maximum)
        self.fluidDataTypeGroupBoxLayout = QtWidgets.QHBoxLayout(self.fluidDataTypeGroupBox)
        self.fluidDataTypeButtonGroup = QtWidgets.QButtonGroup()
        self.fluidDataTypeBinRadioButton = QtWidgets.QRadioButton("Bin")
        self.fluidDataTypeTracerRadioButton = QtWidgets.QRadioButton("Tracer")
        self.fluidDataTypeTracerRadioButton.toggled.connect(self.actionUpdateTracerDataGFCopyButtonEnabled)
        self.fluidDataTypeButtonGroup.addButton(self.fluidDataTypeBinRadioButton, 0)
        self.fluidDataTypeButtonGroup.addButton(self.fluidDataTypeTracerRadioButton, 1)
        self.fluidDataTypeGroupBoxLayout.addWidget(self.fluidDataTypeBinRadioButton)
        self.fluidDataTypeGroupBoxLayout.addWidget(self.fluidDataTypeTracerRadioButton)
        self.fluidDataTypeGroupBox.setLayout(self.fluidDataTypeGroupBoxLayout)
        # Add clear button
        self.clearFluidDataButton = QtWidgets.QPushButton("Clear FluidData from viewer...")
        self.clearFluidDataButton.clicked.connect(self.actionClearFluidData)
        
        #######################################################################
        #######  Group box for controlling display settings
        self.fluidDataViewControlGroupBox = QtWidgets.QGroupBox('Display settings FluidData')
        self.fluidDataViewControlGroupBoxLayout = QtWidgets.QGridLayout()
        self.fluidDataViewControlPercentageSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fluidDataViewControlPercentageSlider.setMinimum(0)
        self.fluidDataViewControlPercentageSlider.setMaximum(20)
        self.fluidDataViewControlPercentageSlider.setValue(20)
        # self.fluidDataViewControlPercentageSlider.setSingleStep(1)
        self.fluidDataViewControlPercentageSlider.setTickInterval(2)
        self.fluidDataViewControlPercentageSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.fluidDataViewControlPercentageSlider.valueChanged.connect(self.actionUpdateFluidDataSliderLabel)
        # self.fluidDataViewControlPercentageSlider.sliderReleased.connect(self.actionUpdateFluidDataDisplay)
        
        self.fluidDataViewControlPercentageSliderDisplay = QtWidgets.QLabel()
        self.actionUpdateFluidDataSliderLabel()
        # self.percentageFluidDataToDisplayOld = self.percentageFluidDataToDisplay
        
        # CheckButton for masking with modelPD
        self.fluidDataMaskPDCheckButton = QtWidgets.QCheckBox()
        self.fluidDataMaskPDCheckButton.setText('Blanking with model STL')
        
        # CheckButton for masking with isValid criteria (if exists)
        self.fluidDataMaskValidCheckButton = QtWidgets.QCheckBox()
        self.fluidDataMaskValidCheckButton.setText('Only valid points')
        
        # CheckButton for using crop dimensions
        self.fluidDataMaskCropCheckButton = QtWidgets.QCheckBox()
        self.fluidDataMaskCropCheckButton.setText('Crop data')
        
        # CheckButton for showing fluid data
        self.fluidDataMaskShowCheckButton = QtWidgets.QCheckBox('Show Fluid data')
        self.fluidDataMaskShowCheckButton.setChecked(True)
        self.fluidDataMaskShowCheckButton.stateChanged.connect(self.showFluidData)
        
        # Pushbutton to update the fluid data view
        self.fluidDataUpdatePushButton = QtWidgets.QPushButton("Update FluidData view")
        self.fluidDataUpdatePushButton.clicked.connect(self.actionUpdateFluidDataDisplay)
        
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataViewControlPercentageSlider, 0, 0, 1, 5)
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataViewControlPercentageSliderDisplay, 0, 5, 1, 1)
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataMaskPDCheckButton, 1, 0, 1, 2)
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataMaskValidCheckButton, 1, 2, 1, 1)
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataMaskCropCheckButton, 1, 3, 1, 1)
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataMaskShowCheckButton, 2, 0, 1, 1)
        self.fluidDataViewControlGroupBoxLayout.addWidget(self.fluidDataUpdatePushButton, 2, 1, 1, 5)
        
        self.fluidDataViewControlGroupBox.setLayout(self.fluidDataViewControlGroupBoxLayout)
        
        # Add widgets to layout
        self.fluidDataGroupBoxLayout.addWidget(self.loadFluidDataDisplayFile, 0, 0, 1, 3)
        self.fluidDataGroupBoxLayout.addWidget(self.chooseFluidDataButton, 0, 3, 1, 1)
        self.fluidDataGroupBoxLayout.addWidget(self.fluidDataTypeGroupBox, 1, 0, 1, 4)
        self.fluidDataGroupBoxLayout.addWidget(self.loadFluidDataButton, 2, 0, 1, 2)
        self.fluidDataGroupBoxLayout.addWidget(self.clearFluidDataButton, 2, 2, 1, 2)
        self.fluidDataGroupBoxLayout.addWidget(self.fluidDataViewControlGroupBox, 3, 0, 1, 4)
        
        # Finally set the layout to the group box
        self.fluidDataGroupBox.setLayout(self.fluidDataGroupBoxLayout)
        
        #######################################################################
        #######  Group box for tracer particle data (for groundPlaneFit estimate)
        self.tracerDataGFGroupBox = QtWidgets.QGroupBox("Define Tracer data [update object registration]")
        self.tracerDataGFGroupBoxLayout = QtWidgets.QGridLayout()
        
        # Add "copy from fluid mesh" button
        self.tracerDataGFCopyButton = QtWidgets.QPushButton('Copy from FluidData')
        self.tracerDataGFCopyButton.clicked.connect(self.actionCopyTracerDataGFFromFluidData)
        self.actionUpdateTracerDataGFCopyButtonEnabled()
        
        # Add display bar
        self.loadTracerDataGFDisplayFile = QtWidgets.QLineEdit()
        self.loadTracerDataGFDisplayFile.readOnly = True
        # Add pushbutton to choose file
        self.chooseTracerDataGFButton = QtWidgets.QPushButton("Choose TracerData file for Ground-Fit")
        self.chooseTracerDataGFButton.clicked.connect(self.actionOpenTracerDataGFFileDialog)
        self.tracerDataGFFileName = ''
        # Add load button
        self.loadTracerDataGFButton = QtWidgets.QPushButton("Load TracerData file for Ground-Fit...")
        self.loadTracerDataGFButton.clicked.connect(self.actionLoadTracerDataGFThreaded)
        # Add clear button
        self.clearTracerDataGFButton = QtWidgets.QPushButton("Clear TracerData from viewer...")
        self.clearTracerDataGFButton.clicked.connect(self.actionClearTracerDataGF)
        
        # Add a checkbox to show or not show tracer data
        self.tracerDataShowCheckBox = QtWidgets.QCheckBox('Show Tracer data')
        self.tracerDataShowCheckBox.setChecked(True)
        self.tracerDataShowCheckBox.stateChanged.connect(self.showTracerData)
        
        # Add widgets to layout
        self.tracerDataGFGroupBoxLayout.addWidget(self.tracerDataGFCopyButton, 0, 0, 1, 4)
        self.tracerDataGFGroupBoxLayout.addWidget(self.loadTracerDataGFDisplayFile, 1, 0, 1, 3)
        self.tracerDataGFGroupBoxLayout.addWidget(self.chooseTracerDataGFButton, 1, 3, 1, 1)
        self.tracerDataGFGroupBoxLayout.addWidget(self.tracerDataShowCheckBox, 2, 0, 1, -1)
        self.tracerDataGFGroupBoxLayout.addWidget(self.loadTracerDataGFButton, 3, 0, 1, 2)
        self.tracerDataGFGroupBoxLayout.addWidget(self.clearTracerDataGFButton, 3, 2, 1, 2)
        
        # Finally set the layout to the group box
        self.tracerDataGFGroupBox.setLayout(self.tracerDataGFGroupBoxLayout)
        
        self.controllerFluidGroupBoxLayout.addWidget(self.fluidDataGroupBox)
        self.controllerFluidGroupBoxLayout.addWidget(self.tracerDataGFGroupBox)
        self.controllerFluidGroupBox.setLayout(self.controllerFluidGroupBoxLayout)
        
        #######################################################################
        #######  Define layout of Controller
        # Add groupbox widgets to the layout
        self.controllerWidgetGeometryLayout.addWidget(self.controllerGeometryGroupBox)
        # self.controllerLayout.addWidget(self.outputMeshGroupBox)
        self.controllerWidgetFluidLayout.addWidget(self.controllerFluidGroupBox)
        # self.controllerLayout.addWidget(self.tracerDataGFGroupBox)
        
        # Set the layout of the controller widget
        self.controllerWidgetGeometry.setLayout(self.controllerWidgetGeometryLayout)
        self.controllerWidgetFluid.setLayout(self.controllerWidgetFluidLayout)
        
        # Define the starting settings
        self.blankPointsInsideModelPD = False
        return
    
    def actionCreateOutputMeshFromSTL(self):
        if self.groundPlaneExists:
            # Triangulate the plane first
            planeTriangles = vtk.vtkTriangleFilter()
            planeTriangles.SetInputConnection(self.modelPlane.GetOutputPort())
            planeTriangles.Update()
            
            # Use pyvista to simplify boolean operation
            meshStl = pv.wrap(self.modelSTLreader.GetOutput())
            meshPlane = pv.wrap(planeTriangles.GetOutput())
            
            # Compute boolean interection
            inputMesh = meshStl.boolean_union(meshPlane)
                    
        else:
            inputMesh = self.modelSTLreader.GetOutput()
            
        
        ## Create a mesh using subdivision filter
        ASF = vtk.vtkAdaptiveSubdivisionFilter()
        ASF.SetInputData(inputMesh)
        ASF.SetMaximumEdgeLength(3.)
        ASF.Update()
        
        # Create output meshes
        self.vtkOutputMesh = ASF.GetOutput()
        self.vtkAdaptedOutputMesh = dsa.WrapDataObject(self.vtkOutputMesh)
        
        # Save the info
        self.updateOutputMeshInfo()
        self.updateOutputMeshInfoItems()
        
        self.displayOutputMesh()
        
    def showModelPD(self, val):
        # Check if loaded
        if self.modelPDLoaded:
            self.modelPDActor.SetVisibility(val)
            
            # Update view
            self.vtkWidgetViewer.GetRenderWindow().Render()
            # self.vtkWidgetIren.Initialize()
    
    def showOutputMesh(self, val):
        # Check if loaded
        if self.outputMeshLoaded:
            self.outputMeshActor.SetVisibility(val)
            
            # Update view
            self.vtkWidgetViewer.GetRenderWindow().Render()
            # self.vtkWidgetIren.Initialize()

    def showFluidData(self, val):
        # Check if loaded
        if self.fluidDataLoaded:
            self.fluidDataActor.SetVisibility(val)
            
            # Update view
            self.vtkWidgetViewer.GetRenderWindow().Render()
            # self.vtkWidgetIren.Initialize()
    
    def showTracerData(self, val):
        # Check if loaded
        if self.tracerDataLoaded:
            self.tracerDataGFActor.SetVisibility(val)
            
            # Update view
            self.vtkWidgetViewer.GetRenderWindow().Render()
            # self.vtkWidgetIren.Initialize()
    
    def actionClearTracerDataGF(self):
        if hasattr(self, 'tracerDataGFActor'):
            self.vtkWidgetRen.RemoveActor(self.tracerDataGFActor)
        self.vtkWidgetIren.Initialize()
        
        # Reset the tracer data info
        self.NpTD = ''
        self.NpTDString = ''
        self.updateTracerDataInfoItems()
        self.tracerDataLoaded = False
        
        # Enable settingss
        self.enableSettings()
        
        # Add logging
        logging.info('Tracer Data for ground-fit cleared')
        
    @QtCore.pyqtSlot(object)
    def updateFluidInfo(self, info):
        self.fluidInfo = info
        self.fluidDataNHeaderRows = info['NHeaderLines']
        
        logging.info('Updating fluid info')
        
        # Define datafile title
        self.FMTitle = info['Title']
        self.FMTitleString = self.FMTitle
        
        # Set number of points
        self.NpFM = int(info["NDataLines"])
        self.NpFMString = f'{self.NpFM:,}'
        
        findBins = re.search('[0-9]+x[0-9]+x[0-9]+', self.FMTitle)
        binsSizeString = findBins.group().split('x')
        binsSize = [int(item) for item in binsSizeString]
        self.binSize = int(binsSize[0])
        self.binSizeString = str(self.binSize)
        
        findPer = re.search('_[0-9]+per', self.FMTitle)
        self.overlap = int(findPer.group().lstrip('_').rstrip('per'))
        self.overlapString = str(self.overlap )
        
        self.pitchPX = round(self.binSize * (1 - self.overlap/100))
        self.pitchPXString = str(self.pitchPX)
        
        # Update the list
        self.updateFluidMeshInfoItems()
        
    def updateOutputMeshInfo(self):
        # We extract the number of outputmesh points ...
        self.NpOut = self.vtkOutputMesh.GetNumberOfPoints()
        self.NpOutString = f'{self.vtkOutputMesh.GetNumberOfPoints():,}'
        # and the number of cells'
        self.NcOut = self.vtkOutputMesh.GetNumberOfCells()
        self.NcOutString = f'{self.vtkOutputMesh.GetNumberOfCells():,}'
        
        # Set maximum of evaluation box
        self.singleEvaluationIdx.setMaximum(self.NcOut)
    
    def updateObjectMeshInfo(self):
        # We extract the number of object (STL) points ...
        self.NpObject = self.modelSTLreader.GetOutput().GetNumberOfPoints()
        self.NpObjectString = f'{self.modelSTLreader.GetOutput().GetNumberOfPoints():,}'
        # and the number of cells
        self.NcObject = self.modelSTLreader.GetOutput().GetNumberOfCells()
        self.NcObjectString = f'{self.modelSTLreader.GetOutput().GetNumberOfCells():,}'
        # Flag for ground plane
        self.GP = int(self.groundPlaneExists)
        self.GPString = str(self.groundPlaneExists)
        
    def updateTracerDataInfo(self):
        # Extract the number of tracer data points
        self.NpTD = len(self.tracerDataGF)
        self.NpTDString = f'{self.NpTD:,}'
    
    def updateList(self):
        
        # self.logValuesListItemPixelToMM_PXpMM = QtWidgets.QListWidgetItem()
        if self.fluidDataLoaded:
            if self.fluidDataTypeBinRadioButton.isChccked():
                # Determine voxel size based on size of Cartesian velocity field grid.
                x_pos_ordered = sorted(set(self.fluidDataVelfield.iloc[:,0]))
                y_pos_ordered = sorted(set(self.fluidDataVelfield.iloc[:,1]))
                z_pos_ordered = sorted(set(self.fluidDataVelfield.iloc[:,2]))
                
                dx = np.array(x_pos_ordered[1:]) - np.array(x_pos_ordered[:-1])
                dy = np.array(y_pos_ordered[1:]) - np.array(y_pos_ordered[:-1])
                dz = np.array(z_pos_ordered[1:]) - np.array(z_pos_ordered[:-1])
                
                logging.info(f'inter-x distance: {np.mean(dx):.2f}, inter-y distance: {np.mean(dy):.2f}, inter-z distance: {np.mean(dz):.2f}')
                self.voxelSize = np.mean([np.mean(dx), np.mean(dy), np.mean(dz)])
                self.fluidData_PXpMM
                
            else:
                self.fluidData_PXpMM = -1
            
            self.fluidData_PXpMM = 0
            self.fluidData_PXpMM = 0
    
    def actionLoadTracerDataGFThreaded(self):
        if self.tracerDataGFFileName == '':
            return
        
        # Add logging
        logging.info('Loading tracer data for ground plane fit...')
        
        # Disable buttons
        self.tracerDataGFCopyButton.setEnabled(False)
        self.chooseTracerDataGFButton.setEnabled(False)
        self.loadTracerDataGFButton.setEnabled(False)
        self.clearTracerDataGFButton.setEnabled(False)
        self.startProgressBar('Loading TracerData for Ground-Fit...')
        
        # Clear previous fluid data
        self.actionClearTracerDataGF()        
        
        # Create a thread do load the fluid data
        self.threadLoadTracerDataGF = LoadFluidDataWorker()
        self.threadLoadTracerDataGF.progress.connect(self.updateProgressBar)
        self.threadLoadTracerDataGF.logSignal.connect(self.logThreaded)
        self.threadLoadTracerDataGF.finishedTracers.connect(self.actionDisplayTracerDataGF)
        self.threadLoadTracerDataGF.execute(self.tracerDataGFFileName,
                                            False)
    
    def actionOpenTracerDataGFFileDialog(self):
        # Pre-determine the folder to open
        folderGuess = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks'
        
        # Open file dialog
        self.tracerDataGFFileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose TracerData file for Ground-Fit', 
                                                                          folderGuess, "TracerData files (*.npy)")
        # Enter filename in display bar
        self.loadTracerDataGFDisplayFile.setText(self.tracerDataGFFileName)
        
        return
    
    def actionCopyTracerDataGFFromFluidData(self):
        # We check if we can actually copy from fluid data mesh
        # (Conditions: Must be loaded and must be tracer-data)
        if (not self.fluidDataLoaded) and (not self.fluidDataTypeTracerRadioButton.isChecked()):
            logging.info('Incorrect settings to copy fluid mesh data for ground plane fit. '
                         'Check if fluid mesh data is loaded and is of type "Tracers"')
            return
        
        # Otherwise we copy everything
        self.tracerDataGFFileName = self.fluidDataFileName
        self.loadTracerDataGFDisplayFile.setText(self.tracerDataGFFileName)
        
        # Display everything through copying
        self.actionDisplayTracerDataGF(self.fluidDataVelfield)
    
    def actionDisplayTracerDataGF(self, loadedData):
        
        # Add logging
        logging.info('Creating mappers and actors to display tracer data for ground fit')
        
        # Save the loaded data attribute
        self.tracerDataGF  = loadedData.iloc[:,:3]
        self.tracerDataGFPD = self._numpy_points_to_vtk_polydata(self.tracerDataGF.to_numpy())
        self.NTracerDataGFPoints = self.tracerDataGFPD.GetNumberOfPoints()
        
        # Set flag to True
        self.tracerDataLoaded = True
        
        # Update the fluid data view
        self.updateTracerDataView()
            
        # Initialise the interactive renderer
        self.vtkWidgetRen.ResetCamera()
        self.showTracerData(self.tracerDataShowCheckBox.isChecked())
        # self.vtkWidgetViewer.GetRenderWindow().Render()
        # self.vtkWidgetIren.Initialize()
        
        # Update the log info items
        self.updateTracerDataInfo()
        self.updateTracerDataInfoItems()
        
        # Re-enable buttons
        self.tracerDataGFCopyButton.setEnabled(True)
        self.chooseTracerDataGFButton.setEnabled(True)
        self.loadTracerDataGFButton.setEnabled(True)
        self.clearTracerDataGFButton.setEnabled(True)
        self.resetProgressBar()
        
        
        # Check if settings can be enabled
        self.enableSettings()
        self.updateRadiusSizeLabel()
        
        # Add logging
        logging.info('Done loading ground-plane tracer data')
        
    def updateTracerDataView(self):
        # Create a selection to display
        selectionTracerDataGF = self.createVTKTracerDataGFSelectionToDisplay()
        
        # Extract using this selection
        extractSelectionTracerDataGF = vtk.vtkExtractSelection()
        extractSelectionTracerDataGF.SetInputData(0, self.tracerDataGFPD)
        extractSelectionTracerDataGF.SetInputData(1, selectionTracerDataGF)
        extractSelectionTracerDataGF.Update()
                
        # Create a mapper for the fluid data
        tracerDataGFMapper = vtk.vtkDataSetMapper()
        tracerDataGFMapper.SetInputData(extractSelectionTracerDataGF.GetOutput())
        
        # Create actor for the fluid data
        self.tracerDataGFActor = vtk.vtkActor()
        self.tracerDataGFActor.SetMapper(tracerDataGFMapper)
        self.tracerDataGFActor.GetProperty().SetColor(self.vtkColors.GetColor3d("DarkGray"))
        self.tracerDataGFActor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.vtkWidgetRen.AddActor(self.tracerDataGFActor)
    
    def actionUpdateTracerDataGFCopyButtonEnabled(self):
        if self.fluidDataTypeTracerRadioButton.isChecked():
            self.tracerDataGFCopyButton.setEnabled(True)
        else:
            self.tracerDataGFCopyButton.setEnabled(False)
    
    def actionUpdateFluidDataSliderLabel(self):
        self.percentageFluidDataToDisplay = self.fluidDataViewControlPercentageSlider.value() * 5
        self.fluidDataViewControlPercentageSliderDisplay.setText(str(self.percentageFluidDataToDisplay) + ' %')
        return
    
    def actionUpdateFluidDataDisplay(self):
        # Check if model data is loaded
        if not self.fluidDataLoaded:
            return
            
        # Add logging
        logging.info(f'Updating view with {self.percentageFluidDataToDisplay}% of FluidData points.')
        
        self.startSpinner()
        
        # Disable the slider
        self.fluidDataViewControlPercentageSlider.setEnabled(False)
        
        # Remove the old actor
        self.vtkWidgetRen.RemoveActor(self.fluidDataActor)
        
        # Update the fluid data view using a separate thread
        self.updateFluidDataView()
        
        # Re-initialize the interactive renderer
        self.vtkWidgetIren.Initialize()
        
        # Re-enable the slider
        self.fluidDataViewControlPercentageSlider.setEnabled(True)
        self.stopSpinner()
    
    def actionLoadModelPD(self):
        # Check if we can proceed
        if self.modelPDFileName == '':
            return
        
        logging.info('Loading STL file...')
        
        # Clear old actors
        self.actionClearModelPD()
        
        # Open STl file and show in vtkWidgetRenderer
        self.modelSTLreader = vtk.vtkSTLReader()
        self.modelSTLreader.SetFileName(self.modelPDFileName)
        self.modelSTLreader.Update()
        
        # Create mappers
        modelSTLMapper = vtk.vtkPolyDataMapper()
        modelSTLMapper.SetInputConnection(self.modelSTLreader.GetOutputPort())
        
        # Create an actor
        self.modelPDActor = vtk.vtkActor()
        self.modelPDActor.SetMapper(modelSTLMapper)
        self.modelPDActor.GetProperty().SetColor(self.vtkColors.GetColor3d("Navy"))
        
        # Add actor to renderer
        self.vtkWidgetRen.AddActor(self.modelPDActor)
        
        ################
        # Now do that for a plane too
        if self.groundPlaneCheckBox.isChecked():
            logging.info('Adding horizontal ground plane')
            self.modelPlane = vtk.vtkPlaneSource()
            self.modelPlane.SetPoint1(500.0, 0.0, 0.0)
            self.modelPlane.SetPoint2(0.0, 500.0, 0.0)
            self.modelPlane.SetCenter(0.0, 0.0, self.groundPlaneZPosBox.value())
            self.modelPlane.Update()
            self.groundPlaneExists = True

            # Create mappers
            modelPlaneMapper = vtk.vtkPolyDataMapper()
            modelPlaneMapper.SetInputConnection(self.modelPlane.GetOutputPort())
            
            # Create an actor
            self.modelPlaneActor = vtk.vtkActor()
            self.modelPlaneActor.SetMapper(modelPlaneMapper)
            
            if self.showGroundPlaneCorrectedCheckButton.isChecked() and self.GPFitLoaded:
                self.modelPlaneActor.GetProperty().SetRepresentationToWireframe()
                self.modelPlaneActor.GetProperty().SetLineWidth(3)
            else:
                self.modelPlaneActor.GetProperty().SetColor(self.vtkColors.GetColor3d("Navy"))
                self.modelPlaneActor.GetProperty().SetRepresentationToSurface()
            # Add actor to renderer
            self.vtkWidgetRen.AddActor(self.modelPlaneActor)
        else:
            self.groundPlaneExists = False
        
        
        # Update the information
        self.updateObjectMeshInfo()
        self.updateObjectMeshInfoItems()
 
        # Set the flag
        self.modelPDLoaded = True
        self.createOutputMeshFromSTLButton.setEnabled(True)
        
        # Reset camera angle
        self.vtkWidgetRen.ResetCamera()
        self.showModelPD(self.modelPDShowCheckButton.isChecked())
        # # Initialise the interactive renderer
        # self.vtkWidgetViewer.GetRenderWindow().Render()
        # self.vtkWidgetIren.Initialize()
        
        # Check if settings can be enabled
        self.enableSettings()
        
        # Add logging
        logging.info('STL file loaded')
        return
 
    def actionClearModelPD(self):
        if hasattr(self, 'modelPDActor'):
            self.vtkWidgetRen.RemoveActor(self.modelPDActor)
        if hasattr(self, 'modelPlaneActor'):
            self.vtkWidgetRen.RemoveActor(self.modelPlaneActor)
        self.vtkWidgetIren.Initialize()
        
        # Reset the info items
        self.NcObject = ''
        self.NcObjectString = ''
        self.NpObject = ''
        self.NpObjectString = ''
        self.GP = ''
        self.GPString = ''
        self.updateObjectMeshInfoItems()
        self.modelPDLoaded = False
        self.createOutputMeshFromSTLButton.setEnabled(False)
        
        # Enable settingss
        self.enableSettings()
        
        # Add logging
        logging.info('STL data cleared')
        return
    
    def actionOpenSaveDirectoryDialog(self):
        # Open file dialog
        # ADD THIS FOLDER TO THE SETTINGS
        self.saveDirectory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose save-folder', 
         r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\results',)
         # options=QtWidgets.QFileDialog.ShowDirsOnly)
        
        # Enter directory name in display bar
        self.saveDirectoryLine.setText(self.saveDirectory)
    
    def actionOpenGroundPlaneFitFolderDialog(self):
        # Open file dialog
        self.groundPlaneFitFolder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose ground plane fit folder', 
         r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data',)
         # options=QtWidgets.QFileDialog.ShowDirsOnly)
        
        # Enter directory name in display bar
        self.groundPlaneFitFolderLine.setText(self.groundPlaneFitFolder)
    
    def actionOpenModelPDFileDialog(self):
        # Open file dialog
        self.modelPDFileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose object STL-file', 
         r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Erik_cube_binning_32x32x32',"STL files (*.stl)")
        
        # Enter filename in display bar
        self.loadModelPDDisplayFile.setText(self.modelPDFileName)
        return
        
    def actionLoadOutputMesh(self):
        # Check if we can proceed
        if self.outputMeshFileName == '':
            return
        
        # Clear old actors
        self.actionClearOutputMesh()
        
        # Add logging
        logging.info('Loading VTK OutputMesh file...')
        
        # Load the vtk mesh using the pivDA functions
        outputMeshDirty, _ = pivDA.load_unstructured_vtk_grid(self.outputMeshFileName)

        # Clean up meshes
        self.vtkOutputMesh, self.vtkAdaptedOutputMesh = pivDA.vtkmesh_cleanup(outputMeshDirty, vtk.VTK_TRIANGLE)
        
        # Save the info
        self.updateOutputMeshInfo()
        self.updateOutputMeshInfoItems()
        
        # Display OutputMesh
        self.displayOutputMesh()
        
    def displayOutputMesh(self):
        # Add logging
        logging.info('Creating mappers and actors to display VTK OutputMesh')
        # Create a mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.vtkOutputMesh)
        
        # Create an actor
        self.outputMeshActor = vtk.vtkActor()
        self.outputMeshActor.SetMapper(mapper)
        self.outputMeshActor.GetProperty().SetRepresentationToWireframe()
        self.outputMeshActor.GetProperty().SetColor(self.vtkColors.GetColor3d("DarkSalmon"))
        self.outputMeshActor.GetProperty().SetLineWidth(3)
        
        # Add actor to renderer
        self.vtkWidgetRen.AddActor(self.outputMeshActor)
            
        # Set flag
        self.outputMeshLoaded = True
        
        # Initialise the interactive renderer
        self.vtkWidgetRen.ResetCamera()
        self.showOutputMesh(self.outputMeshShowCheckButton.isChecked())
        
        # Update the style to include selecting the outputmesh
        self.vtkWidgetRenStyle = MouseInteractorStyle(self.vtkOutputMesh,
                                                      self.singleEvaluationIdx.setValue,
                                                      self.outputMeshActor)
        self.vtkWidgetRenStyle.SetDefaultRenderer(self.vtkWidgetRen)
        self.vtkWidgetIren.SetInteractorStyle(self.vtkWidgetRenStyle)
        
        # self.vtkWidgetViewer.GetRenderWindow().Render()
        # self.vtkWidgetIren.Initialize()
        
        # Check if settings can be enabled
        self.enableSettings()
        
        # Add logging
        logging.info('OutputMesh loaded')
        return
    
    def actionClearOutputMesh(self):
        if hasattr(self, 'outputMeshActor'):
            self.vtkWidgetRen.RemoveActor(self.outputMeshActor)
        self.vtkWidgetIren.Initialize()
        
        # Clear the info items
        self.NpOut = ''
        self.NpOutString = ''
        self.NcOut = ''
        self.NcOutString = ''
        self.updateOutputMeshInfoItems()
        self.outputMeshLoaded = False
        self.singleCellPreviouslyLoaded = False
        
        # Enable settingss
        self.enableSettings()
        
        # Add logging
        logging.info('OutputMesh cleared')
        return
    
    def actionOpenOutputMeshFileDialog(self):
        # Open file dialog
        self.outputMeshFileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose OutputMesh VTK-file', 
         r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes', "VTK files (*.vtk)")
        
        # Enter filename in display bar
        self.loadOutputMeshDisplayFile.setText(self.outputMeshFileName)
        return
    
    def actionOpenFluidDataFileDialog(self):
        # Pre-determine the folder to open
        if self.fluidDataTypeTracerRadioButton.isChecked():
            folderGuess = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks"
        elif self.fluidDataTypeBinRadioButton.isChecked():
            folderGuess = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Erik_cube_binning_32x32x32\TecPlot\data-files'
        else:
            folderGuess = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data'
        
        # Open file dialog
        self.fluidDataFileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose FluidData file', 
         folderGuess, "FluidData files (*.npy *.dat)")
        
        # Enter filename in display bar
        self.loadFluidDataDisplayFile.setText(self.fluidDataFileName)
        
        # Set the radiobutton based on the extension
        if self.fluidDataFileName != '':
            pathlibFluidDataFileName = Path(self.fluidDataFileName)
            if pathlibFluidDataFileName.suffix == '.npy':
                self.fluidDataTypeTracerRadioButton.setChecked(True)
            elif pathlibFluidDataFileName.suffix == '.dat':
                self.fluidDataTypeBinRadioButton.setChecked(True)
        return
    
    def actionLoadFluidDataThreaded(self):
        if self.fluidDataFileName == '':
            return
        
        # Add logging
        logging.info('Loading fluid data...')
        
        # Disable buttons
        self.fluidDataGroupBox.setEnabled(False)
        self.startProgressBar('Loading FluidData...')
        
        # Clear previous fluid data
        self.actionClearFluidData()
        
        # Create a thread do load the fluid data
        self.threadLoadFluidData = LoadFluidDataWorker()
        self.threadLoadFluidData.progress.connect(self.updateProgressBar)
        self.threadLoadFluidData.logSignal.connect(self.logThreaded)
        self.threadLoadFluidData.headerDataSignal.connect(self.updateFluidInfo)
        
        if self.fluidDataTypeBinRadioButton.isChecked():
            self.fluidDataTypeIsBin = True
            self.threadLoadFluidData.finishedBin.connect(self.actionDisplayFluidData)
        else:
            self.fluidDataTypeIsBin = False
            self.threadLoadFluidData.finishedTracers.connect(self.actionDisplayFluidData)
        self.threadLoadFluidData.execute(self.fluidDataFileName,
                                         self.fluidDataTypeBinRadioButton.isChecked())
        
        return
    
    def createVTKTracerDataGFSelectionToDisplay(self):
        # 1. Determine the IDs to display
        # Create a mask for the model loaded points
        if self.modelPDLoaded:
            _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(self.tracerDataGF.iloc[:,:3].to_numpy(),
                                                                   self.modelSTLreader.GetOutput()
                                                                   )

            modelMaskPointsToDisplay = (maskInsideSTL == 0)
        else:
            modelMaskPointsToDisplay = np.ones_like(self.tracerDataGF.iloc[:,0].to_numpy(),
                                                    dtype=bool)

        idsAfterMasks = self.tracerDataGF.index.to_numpy()[modelMaskPointsToDisplay]
        
        # Sample 1% of points for display
        random.seed(0)
        NPointsToPlot = int(round(0.01 * len(idsAfterMasks)))
        idsToDisplay = sorted(random.sample(list(idsAfterMasks), NPointsToPlot))
        logging.debug(f'Plotting {NPointsToPlot:,}/{self.NTracerDataGFPoints:,} '
                      f'TracerData points with modelMask: {self.modelPDLoaded}')
        
        # Create a selection
        idsVTK = numpy_support.numpy_to_vtk(idsToDisplay)
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(idsVTK)
        
        # Create a vtkSelection instance
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        
        return selection
    
    def createVTKFluidDataSelectionToDisplay(self):
        # 1. Determine the IDs to display
        # Create a mask for the model loaded points
        if self.modelPDLoaded and self.fluidDataMaskPDCheckButton.isChecked():
            _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(self.fluidDataVelfield.iloc[:,:3].to_numpy(),
                                                                   self.modelSTLreader.GetOutput())
            # maskAboveGround = self.fluidDataVelfield.iloc[:,2].to_numpy() >= 0
            
            modelMaskPointsToDisplay = (maskInsideSTL == 0)# &
                                       # maskAboveGround)
        else:
            modelMaskPointsToDisplay = np.ones_like(self.fluidDataVelfield.iloc[:,0].to_numpy(),
                                                    dtype=bool)
            
        # Create a mask for the valid points
        if self.fluidDataMaskValidCheckButton.isChecked() and ('isValid' in self.fluidDataVelfield):
            maskValidPointsValid = self.fluidDataVelfield['isValid']
        else:
            maskValidPointsValid = np.ones_like(self.fluidDataVelfield.iloc[:,0].to_numpy(),
                                                dtype=bool)

        # Create a mask based on the valid points
        if self.fluidDataMaskCropCheckButton.isChecked() and self.modelPDLoaded:
            # 1. Find the lower limit of cropping
            # a. Compute the lower offset
            if self.groundPlaneExists:
                offsetLower = self.rangeCropVelfieldMin.value() - self.groundPlaneZPosBox.value()
            else:
                offsetLower = self.rangeCropVelfieldMin.value()
                
            # b. Crop lower limit
            if offsetLower > 0:
                maskCropPointsLower = pivDA.mask_by_offset_to_stl(self.fluidDataVelfield.iloc[:,:3].to_numpy(),
                                                                  self.modelSTLreader.GetFileName(),
                                                                  offset = offsetLower,
                                                                  offsetGP = self.groundPlaneExists
                                                                  )
            else:
               maskCropPointsLower = pivDA.maskWithOffsetToSTL(self.fluidDataVelfield.iloc[:,:3].to_numpy(),
                                                               self.modelSTLreader.GetFileName(),
                                                               offset = np.abs(offsetLower),
                                                               offsetGP = self.groundPlaneExists
                                                               )
                
            # 2. Find the upper limit of cropping
            # a. Compute the upper offset
            if self.groundPlaneExists:
                offsetUpper = self.rangeCropVelfieldMax.value() - self.groundPlaneZPosBox.value()
            else:
                offsetUpper = self.rangeCropVelfieldMax.value()
                
            # b. Crop lower limit
            if offsetUpper > 0:
               maskCropPointsHigher = pivDA.mask_by_offset_to_stl(self.fluidDataVelfield.iloc[:,:3].to_numpy(),
                                                                  self.modelSTLreader.GetFileName(),
                                                                  offset = offsetUpper,
                                                                  offsetGP = self.groundPlaneExists
                                                                  )
            else:
               maskCropPointsHigher = pivDA.maskWithOffsetToSTL(self.fluidDataVelfield.iloc[:,:3].to_numpy(),
                                                                self.modelSTLreader.GetFileName(),
                                                                offset = np.abs(offsetUpper),
                                                                offsetGP = self.groundPlaneExists
                                                                )
            
            # 3. Combine the lower and upper crop limits 
            maskCropPoints = (maskCropPointsLower == 1) & (maskCropPointsHigher == 0)            
            
        else:
            # If no cropping, use a matrix of only True values
            maskCropPoints = np.ones_like(self.fluidDataVelfield.iloc[:,0].to_numpy(),
                                          dtype=bool)

        idsAfterMasks = self.fluidDataVelfield.index.to_numpy()[(modelMaskPointsToDisplay) &
                                                                (maskValidPointsValid) &
                                                                (maskCropPoints)]
        
        # Now slice depending on the percentage to display
        
        # Sample number of points depending on the percentage to display
        if self.percentageFluidDataToDisplay == 100:
            NPointsToPlot = len(idsAfterMasks)
            idsToDisplay = idsAfterMasks
        else:
            random.seed(0)
            NPointsToPlot = int(round(self.percentageFluidDataToDisplay / 100 * len(idsAfterMasks)))
            idsToDisplay = sorted(random.sample(list(idsAfterMasks), NPointsToPlot))
        logging.debug(f'Plotting {NPointsToPlot:,}/{self.NFluidDataPoints:,} '
                      ' FluidData points with modelMask: '
                      f'{self.modelPDLoaded and self.fluidDataMaskPDCheckButton.isChecked()}')
        
        # Create a selection
        idsVTK = numpy_support.numpy_to_vtk(idsToDisplay)
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(idsVTK)
        
        # Create a vtkSelection instance
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        
        return selection
    
    def updateFluidDataView(self):
        # Create a selection to display
        self.selection = self.createVTKFluidDataSelectionToDisplay()
        
        # Extract using this selection
        self.extractSelection = vtk.vtkExtractSelection()
        self.extractSelection.SetInputData(0, self.fluidDataPD)
        self.extractSelection.SetInputData(1, self.selection)
        self.extractSelection.Update()
        
        self.selected = self.extractSelection.GetOutput()
        # logging.debug(f'Found {self.selected.GetNumberOfPoints()} points selected in the vtk object')
        # logging.debug(f'Found {self.selected.GetNumberOfCells()} cells selected in the vtk object')
        
        # Create a mapper for the fluid data
        # fluidDataMapper = vtk.vtkPolyDataMapper()
        # fluidDataMapper.SetInputData(self.fluidDataPD)
        fluidDataMapper = vtk.vtkDataSetMapper()
        fluidDataMapper.SetInputData(self.selected)
        
        # Create actor for the fluid data
        self.fluidDataActor = vtk.vtkActor()
        self.fluidDataActor.SetMapper(fluidDataMapper)
        self.fluidDataActor.GetProperty().SetColor(self.vtkColors.GetColor3d("DeepPink"))
        self.fluidDataActor.GetProperty().SetOpacity(0.7)
        
        # Add actor to renderer
        self.vtkWidgetRen.AddActor(self.fluidDataActor)
        
    def enableSettings(self):
        # Disable all methods
        self.mainSettingsMethodNumberGroupBoxButtonJux.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit.setEnabled(False)
        self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth.setEnabled(False)

        # Check if model, outputmesh and fluiddata are loaded
        if self.modelPDLoaded and self.outputMeshLoaded and self.fluidDataLoaded:
            if self.fluidDataTypeIsBin:
                # Enable the bin-based methods
                self.mainSettingsMethodNumberGroupBoxButtonJux.setEnabled(True)
                self.mainSettingsMethodNumberGroupBoxButtonBinConLinInterp.setEnabled(True)
                self.mainSettingsMethodNumberGroupBoxButtonBinNudgeQuadFit.setEnabled(True)
            else:
                # Enable the tracer-based methods
                self.mainSettingsMethodNumberGroupBoxButtonTracerNudgeQuadFit.setEnabled(True)
                self.mainSettingsMethodNumberGroupBoxButtonTracerConQuadFit.setEnabled(True)
                self.mainSettingsMethodNumberGroupBoxButtonTracerGroundTruth.setEnabled(True)
                
        
            
    @QtCore.pyqtSlot(pd.DataFrame)
    def actionDisplayFluidData(self, loadedData):
        # Add logging
        logging.info('Creating mappers and actors to display fluid data')
        
        # Save the loaded data attribute
        self.fluidDataVelfield  = loadedData
        self.fluidDataPD = self._numpy_points_to_vtk_polydata(self.fluidDataVelfield.iloc[:,:3].to_numpy())
        self.NFluidDataPoints = self.fluidDataPD.GetNumberOfPoints()
        
        # Update the fluid data view
        self.updateFluidDataView()
        
        # Set flag to True
        self.fluidDataLoaded = True
            
        # Initialise the interactive renderer
        self.vtkWidgetRen.ResetCamera()       
        self.showFluidData(self.fluidDataMaskShowCheckButton.isChecked())
        # self.vtkWidgetViewer.GetRenderWindow().Render()
        # self.vtkWidgetIren.Initialize()
        
        # Re-enable buttons
        self.fluidDataGroupBox.setEnabled(True)
        self.resetProgressBar()
        
        if self.fluidDataTypeIsBin:
            ###### Update the final items in the list
            x_pos = set(self.fluidDataVelfield.iloc[:,0])
            x_pos_ordered = sorted(x_pos)
            
            y_pos = set(self.fluidDataVelfield.iloc[:,1])
            y_pos_ordered = sorted(y_pos)
            
            z_pos = set(self.fluidDataVelfield.iloc[:,2])
            z_pos_ordered = sorted(z_pos)
            
            dx = np.array(x_pos_ordered[1:]) - np.array(x_pos_ordered[:-1])
            dy = np.array(y_pos_ordered[1:]) - np.array(y_pos_ordered[:-1])
            dz = np.array(z_pos_ordered[1:]) - np.array(z_pos_ordered[:-1])
            
            # Set pixel to mm
            self.pitchMM = np.round(np.mean([np.mean(dx), np.mean(dy), np.mean(dz)]), 3)
            self.pitchMMString = str(self.pitchMM)
            
            self.PXpMM = np.round(self.pitchPX / self.pitchMM, 3)
            self.PXpMMString = str(self.PXpMM)
            
            # Update the data in the list
            self.updateFluidMeshInfoItems()
            
        # Check if settings can be enabled
        self.enableSettings()
        self.updateRadiusSizeLabel()
        
        # Add logging
        logging.info('Done loading fluid data')

    def actionClearFluidData(self):
        if hasattr(self, 'fluidDataActor'):
            self.vtkWidgetRen.RemoveActor(self.fluidDataActor)
        else:
            logging.debug('No fluid data to clear')
            return
        # Re-initialize the interactive window
        self.vtkWidgetIren.Initialize()
        
        # Clear the fluid data info
        self.FMTitle = ''
        self.FMTitleString = ''
        self.binSize = ''
        self.binSizeString = ''
        self.overlap = ''
        self.overlapString = ''
        self.pitchMM = ''
        self.pitchMMString = ''
        self.pitchPX = ''
        self.pitchPXString = ''
        self.PXpMM = ''
        self.PXpMMString = ''
        self.NpFMS = ''
        self.NpFMString = ''
        self.updateFluidMeshInfoItems()
        self.fluidDataLoaded = False
        
        # Enable settingss
        self.enableSettings()
        
        # Add logging
        logging.info('Fluid data cleared')
        return
        
    def CreateRightMenu(self):
        pass

    def _numpy_points_to_vtk_polydata(self, points):
        # Cast points to check into vtk points object
        vtkPoints = vtk.vtkPoints()
        vtkCells = vtk.vtkCellArray()
        N = len(points)
        
        ###########
        # Convert from numpy to vtk (Add deep is true, such that numpy array can be gc'd)
        points_conv = numpy_support.numpy_to_vtk(points, deep=True)
        
        # Add to vtkPoints object
        vtkPoints.SetNumberOfPoints(N)
        vtkPoints.SetData(points_conv)
        
        ##########
        # Create connectivity array for 1-dimensional cells
        cells_conn = numpy_support.numpy_to_vtkIdTypeArray(np.arange(N, dtype=np.int64), deep=True)
        
        # Add to vtkCellArray object
        CellLength = 1
        vtkCells.SetData(CellLength, cells_conn)
        
        # for point in points:
        #     pointId = vtkPoints.InsertNextPoint(point[:])
        #     vtkCells.InsertNextCell(1)
        #     vtkCells.InsertCellPoint(pointId)

        # Cast points into vtk polydata object
        vtkPolydata = vtk.vtkPolyData()
        vtkPolydata.SetPoints(vtkPoints)
        vtkPolydata.SetVerts(vtkCells)

        return vtkPolydata
 
if __name__ == "__main__":
 
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    # Create the window
    window = MainWindow()
    window.setup = False
    
    # Create the setup dialog
    setupdialog = ProgressBarSetupDialog()
    
    if setupdialog.exec_() == QtWidgets.QDialog.Accepted:
        window.setup = True
        window.show()
    else:
        window.close()
        sys.exit(0)
 
    # app.exec_()
    sys.exit(app.exec_())
    
