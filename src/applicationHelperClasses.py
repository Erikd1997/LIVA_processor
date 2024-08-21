# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:41:08 2024

@author: ErikD
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import vtk
import keyboard as kb
import numpy as np
import logging

class QSettingsWindow(QtWidgets.QMainWindow):
    okPressed = QtCore.pyqtSignal()
    cancelPressed = QtCore.pyqtSignal()
    applyPressed = QtCore.pyqtSignal()
    def __init__(self, parent = None, relative_size = (0.6, 0.6), name = None):
        super(QSettingsWindow, self).__init__(parent)
        
        if not isinstance(parent, type(None)):
            size = (int(relative_size[0] * self.parentWidget().width()),
                    int(relative_size[1] * self.parentWidget().height()))
        else:
            size = (400, 400)
            
        self.resize(*size)
        
        if not isinstance(name, type(None)):
            self.setWindowTitle(name)
            
        
        ### Create the main widget layout
        # There is a list widget on the left
        self.tabs_list = QtWidgets.QListWidget()
        self.tabs_list.itemClicked.connect(self._changeWidget)
        self.tabs_list.setMaximumWidth(150)
        
        # There is a stacked widget shown next to that
        self.display_widget = QtWidgets.QStackedWidget()
        
        self.dynamic_widget = QtWidgets.QWidget()
        self.dynamic_widget_layout = QtWidgets.QHBoxLayout()
        self.dynamic_widget_layout.addWidget(self.tabs_list)
        self.dynamic_widget_layout.addWidget(self.display_widget)
        self.dynamic_widget.setLayout(self.dynamic_widget_layout)
        
        # Add buttons
        self.ok_button = QtWidgets.QPushButton("OK")
        self.ok_button.pressed.connect(self.okPressed.emit)
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.apply_button.pressed.connect(self.applyPressed.emit)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.pressed.connect(self.cancelPressed.emit)
        
        self.button_groupbox = QtWidgets.QGroupBox()
        self.button_groupbox_layout = QtWidgets.QHBoxLayout()
        # self.button_groupbox_layout.addWidget(self.apply_button)
        # self.button_groupbox_layout.addWidget(self.cancel_button)
        self.button_groupbox_layout.addWidget(self.ok_button)
        self.button_groupbox.setStyleSheet("QGroupBox {border:0;}")
        self.button_groupbox.setLayout(self.button_groupbox_layout)
        
        # Create the widget that holds these items
        self.main_widget = QtWidgets.QWidget()
        self.main_widget_layout = QtWidgets.QVBoxLayout()
        self.main_widget_layout.addWidget(self.dynamic_widget)
        self.main_widget_layout.addWidget(self.button_groupbox, alignment = QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        
        self.main_widget.setLayout(self.main_widget_layout)
        
        # Add it the the main window
        self.setCentralWidget(self.main_widget)
        
        # Create a dictionary to map the tab name to the widget
        self.mappings = {}
        
    def addEntry(self, entryName, entryWidget, entryIcon=None):
        # Function to add widgets to this view
        if isinstance(entryIcon, type(None)):
            self.tabs_list.addItem(entryName)
        else:
            tab_list_item = QtWidgets.QListWidgetItem(entryIcon, entryName)
            self.tabs_list.addItem(tab_list_item)
        self.display_widget.addWidget(entryWidget)
        
        self.mappings[entryName] = entryWidget
        
        # Slot to change widget based on what is pressed
    def _changeWidget(self, item):
        self.display_widget.setCurrentWidget(self.mappings[item.text()])
        
        # Method to update position of the widget based on the parent's position.
        # Can be called when shown
    def updatePosition(self):        
        relative_loc = (int(self.parentWidget().width() / 2 - self.width() / 2),
                        int(self.parentWidget().height() / 2 - self.height() / 2))
        
        loc = self.parentWidget().geometry().topLeft() + QtCore.QPoint(*relative_loc)
        self.move(loc)

class QtTabWidgetCollapsible(QtWidgets.QTabWidget):
    _contentsVisible = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabBarClicked.connect(self.collapse)

        # Add a button to expand the tabs back
        self.expand_button = QtWidgets.QToolButton(self)
        self.expand_button.setText('<')
        self.expand_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.expand_button.clicked.connect(self.expand)
        self.expand_button.hide()
        
        self.defMaximumWidth(16777215)
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        
        if not self._contentsVisible:
            self.collapse(0)
            
    def moveToCorrectPosition(self):
        if self._contentsVisible:
            # Place the tab in the top right
            loc = self.parent().geometry().topRight() - self.geometry().topRight() - QtCore.QPoint(self.geometry().width(), 0) # - QtCore.QPoint(100, 100) #QtCore.QPoint(self.geometry().width(), 0)
        else:
            loc = self.parent().geometry().topRight() - self.geometry().topRight()

        # self.move(loc)
        # print(f'Moving to {loc}')
            
    def defMaximumWidth(self, max_width):
        self.max_width = max_width
        
        if self._contentsVisible:
            self.setMaximumWidth(max_width)

    def collapse(self, index):
        if index == self.currentIndex():
            self.expand_button.show()
            self.update_expand_button_position()
            
            for i in range(self.count()):
                self.widget(i).setVisible(False)
            
            self.setMaximumWidth(self.tabBar().width() + self.expand_button.width())
            
            self.moveToCorrectPosition()

    def expand(self):
        self.expand_button.hide()
        
        # for i in range(self.count()):
        #     self.widget(i).setVisible(True)
            
        index = self.currentIndex()
        self.widget(index).setVisible(True)
        # self.widget(index).setWidth(self.max_width)
        # print(f'Set widget {index} to visible: {self.widget(index).isVisible()}')
        
        self.setMaximumWidth(self.max_width)  # Reset to default max width
        self.widget(index).setMaximumWidth(self.max_width - self.tabBar().width())
        self.moveToCorrectPosition()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.expand_button.isVisible():
            self.update_expand_button_position()

    def update_expand_button_position(self):
        self.expand_button.move(self.tabBar().width(), 0)
        self.expand_button.resize(self.expand_button.sizeHint())
        
    # def addTab(self, widget, title, *args, **kwargs):
    #     # Adjust size properties
    #     widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        
    #     # Call addTab
    #     super().addTab(widget, title, *args, **kwargs)

class MouseInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, data, func, actorToSearch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AddObserver('LeftButtonPressEvent', self.left_button_press_event)
        self.data = data
        self.func_on_change = func
        self.actor_to_search = actorToSearch
        self.selected_mapper = vtk.vtkDataSetMapper()
        self.selected_actor = vtk.vtkActor()
        
        # Create a vtkCellLocator
        self.data_vtkCellLocator = vtk.vtkCellLocator()
        self.data_vtkCellLocator.SetDataSet(self.data)
        self.data_vtkCellLocator.BuildLocator()

    def left_button_press_event(self, obj, event):
        # Check if control is pressed
        if kb.is_pressed('ctrl+shift'):
            # Get the location of the click (in window coordinates)
            pos = self.GetInteractor().GetEventPosition()
    
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.0005)
    
            # Pick from this location.
            picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())
            
            # Then find the ID corresponding to the desired actor
            actor_id = picker.GetActors().IsItemPresent(self.actor_to_search)
            
            # Then get the picked cell
            if actor_id != 0:
                # Get position from all intersected positions
                picked_position = picker.GetPickedPositions().GetPoint(actor_id-1)
                
                # Find the closest cell
                closest_point = np.array([0.0, 0.0, 0.0])
                cell_obj = vtk.vtkGenericCell()
                projected_cellId_obj = vtk.reference(0)
                subId_obj = vtk.reference(0)
                dist2_obj = vtk.reference(0.)
                
                self.data_vtkCellLocator.FindClosestPoint(picked_position,
                                                          closest_point,
                                                          cell_obj,
                                                          projected_cellId_obj,
                                                          subId_obj,
                                                          dist2_obj
                                                          )
                
                cell_id = projected_cellId_obj
            else:
                cell_id = -1
    
            logging.info(f'Selected cell id: {cell_id}')
            
            self.func_on_change(cell_id)

        # Forward events
        self.OnLeftButtonDown()

class QtTextEditLogger(logging.Handler):
    def __init__(self, parent, *args, **kwargs):
        # Instantiate parent class
        super().__init__(*args, **kwargs)
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                  QtWidgets.QSizePolicy.Preferred)

    def emit(self, record):
        self.msg = self.format(record)
        self.widget.appendPlainText(self.msg)
        
        
class QtWaitingSpinner(QtWidgets.QWidget):
    mColor = QtGui.QColor(QtCore.Qt.gray)
    mRoundness = 100.0
    mMinimumTrailOpacity = 31.4159265358979323846
    mTrailFadePercentage = 50.0
    mRevolutionsPerSecond = 1.57079632679489661923
    mNumberOfLines = 20
    mLineLength = 5
    mLineWidth = 2
    mInnerRadius = 10
    mCurrentCounter = 0
    mIsSpinning = False
    
    # Threaded main settings
    logSignal = QtCore.pyqtSignal(str)
    
    # (self, 
    # roundness=100.0,
    # fade=1.7,
    # radius=27,
    # lines=126,
    # line_length=25,
    # line_width=20,
    # speed=0.86,
    # color=(255, 0, 0)
    # )

    def __init__(self, centerOnParent=False, disableParentWhenSpinning=False, *args, **kwargs):
        super(QtWaitingSpinner, self).__init__()
        self.mCenterOnParent = centerOnParent
        self.mDisableParentWhenSpinning = disableParentWhenSpinning
        self.initialize()

    def initialize(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()

    @QtCore.pyqtSlot()
    def rotate(self):
        self.mCurrentCounter += 1
        if self.mCurrentCounter > self.numberOfLines():
            self.mCurrentCounter = 0
        self.update()

    def updateSize(self):
        size = (self.mInnerRadius + self.mLineLength) * 2
        self.setFixedSize(size, size)

    def updateTimer(self):
        self.timer.setInterval(int(1000 / (self.mNumberOfLines * self.mRevolutionsPerSecond)))

    def updatePosition(self):
        if self.parentWidget() and self.mCenterOnParent:
            self.move(int(self.parentWidget().width() / 2 - self.width() / 2),
                      int(self.parentWidget().height() / 2 - self.height() / 2))

    def lineCountDistanceFromPrimary(self, current, primary, totalNrOfLines):
        distance = primary - current
        if distance < 0:
            distance += totalNrOfLines
        return distance

    def currentLineColor(self, countDistance, totalNrOfLines, trailFadePerc, minOpacity, color):
        if countDistance == 0:
            return color

        minAlphaF = minOpacity / 100.0

        distanceThreshold = np.ceil((totalNrOfLines - 1) * trailFadePerc / 100.0)
        if countDistance > distanceThreshold:
            color.setAlphaF(minAlphaF)

        else:
            alphaDiff = self.mColor.alphaF() - minAlphaF
            gradient = alphaDiff / distanceThreshold + 1.0
            resultAlpha = color.alphaF() - gradient * countDistance
            resultAlpha = min(1.0, max(0.0, resultAlpha))
            color.setAlphaF(resultAlpha)
        return color

    def paintEvent(self, event):
        self.updatePosition()
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self.mCurrentCounter > self.mNumberOfLines:
            self.mCurrentCounter = 0
        painter.setPen(QtCore.Qt.NoPen)

        for i in range(self.mNumberOfLines):
            painter.save()
            painter.translate(self.mInnerRadius + self.mLineLength,
                              self.mInnerRadius + self.mLineLength)
            rotateAngle = 360.0 * i / self.mNumberOfLines
            painter.rotate(rotateAngle)
            painter.translate(self.mInnerRadius, 0)
            distance = self.lineCountDistanceFromPrimary(i, self.mCurrentCounter,
                                                         self.mNumberOfLines)
            color = self.currentLineColor(distance, self.mNumberOfLines,
                                          self.mTrailFadePercentage, self.mMinimumTrailOpacity, self.mColor)
            painter.setBrush(color)
            painter.drawRoundedRect(QtCore.QRect(0, -self.mLineWidth // 2, self.mLineLength, self.mLineLength),
                                    self.mRoundness, QtCore.Qt.RelativeSize)
            painter.restore()
            
    def run(self):
        # Start the spinning
        self.startSpinning()
        
        # Send info
        self.logSignal.emit('Waiting Spinner started')
        
        # Wait till stop button is pressed
        while self.mIsSpinning():
            pass
        
        # Send info
        self.logSignal.emit('Waiting Spinner stopped')
        
    def start(self):
        self.updatePosition()
        self.mIsSpinning = True
        self.show()

        if self.parentWidget() and self.mDisableParentWhenSpinning:
            self.parentWidget().setEnabled(False)

        if not self.timer.isActive():
            self.timer.start()
            self.mCurrentCounter = 0

    def stopSpinning(self):
        self.mIsSpinning = False
        self.hide()

        if self.parentWidget() and self.mDisableParentWhenSpinning:
            self.parentWidget().setEnabled(True)

        if self.timer.isActive():
            self.timer.stop()
            self.mCurrentCounter = 0

    def setNumberOfLines(self, lines):
        self.mNumberOfLines = lines
        self.updateTimer()

    def setLineLength(self, length):
        self.mLineLength = length
        self.updateSize()

    def setLineWidth(self, width):
        self.mLineWidth = width
        self.updateSize()

    def setInnerRadius(self, radius):
        self.mInnerRadius = radius
        self.updateSize()

    def color(self):
        return self.mColor

    def roundness(self):
        return self.mRoundness

    def minimumTrailOpacity(self):
        return self.mMinimumTrailOpacity

    def trailFadePercentage(self):
        return self.mTrailFadePercentage

    def revolutionsPersSecond(self):
        return self.mRevolutionsPerSecond

    def numberOfLines(self):
        return self.mNumberOfLines

    def lineLength(self):
        return self.mLineLength

    def lineWidth(self):
        return self.mLineWidth

    def innerRadius(self):
        return self.mInnerRadius

    def isSpinning(self):
        return self.mIsSpinning

    def setRoundness(self, roundness):
        self.mRoundness = min(0.0, max(100, roundness))

    def setColor(self, color):
        self.mColor = color

    def setRevolutionsPerSecond(self, revolutionsPerSecond):
        self.mRevolutionsPerSecond = revolutionsPerSecond
        self.updateTimer()

    def setTrailFadePercentage(self, trail):
        self.mTrailFadePercentage = trail

    def setMinimumTrailOpacity(self, minimumTrailOpacity):
        self.mMinimumTrailOpacity = minimumTrailOpacity
        
    def decorator(self, func, *args, **kwargs):
        def spinAction(*args, **kwargs):
            self.start()
            func(*args, **kwargs)
            self.stop()
        return spinAction