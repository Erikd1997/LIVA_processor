# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:41:43 2024

Functions to check versions from GitHub repository

@author: ErikD
"""

# Python-core module
import requests
import http.client as httplib
import time
import threading
import os

# Import third party libraries
import sentry_sdk
from packaging.version import Version



# Third party module for geometry manipulation and visualisation
from PyQt5 import QtWidgets, QtCore, QtGui

##############################################
######### FUNCTIONS
##############################################

def get_git_version(owner, repository):
    response = requests.get(f"https://api.github.com/repos/{owner}/{repository}/releases/latest")
    return response.json()["tag_name"]


def have_internet(address: str = "8.8.8.8") -> bool:
    conn = httplib.HTTPSConnection(address, timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()

##############################################
######### CLASSES
##############################################

    
class ProgressBarSetupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initialising LIVA-console")
        self.progressbar = QtWidgets.QProgressBar()
        self.progressbar.setMaximum(100)
        self.label = QtWidgets.QLabel("Initialising LIVA-console")

        button_box = QtWidgets.QDialogButtonBox()
        button_box.setOrientation(QtCore.Qt.Horizontal)
        button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(self.progressbar)
        lay.addWidget(button_box)

        self.setMaximumHeight(self.sizeHint().height())
        self.resize(600, self.height())
        
        #### We connect all settings one-by-one
        setup_checker = SetupChecker()
        
        setup_checker.internCheckStarted.connect(lambda: self.update_label('Polling for internet connection...'))
        setup_checker.internCheckFinished.connect(self.internet_checked)
        
        setup_checker.getCurrentVersionStarted.connect(lambda: self.update_label('Checking current version...'))
        setup_checker.getCurrentVersionFinished.connect(self.current_version_checked)
        
        setup_checker.initSentryStarted.connect(lambda: self.update_label('Initialising cloud error and performance tracing...'))
        setup_checker.initSentryFinished.connect(lambda: self.update_label('Cloud tracer initialised (sentry.io)'))
        
        setup_checker.getLatestVersionStarted.connect(lambda: self.update_label('Checking latest version on Git releases...'))
        setup_checker.getLatestVersionFinished.connect(self.latest_version_checked)
        
        setup_checker.checkVersionsFinished.connect(self.versions_compared)
        
        setup_checker.progress.connect(self.update_progressbar)
        
        setup_checker.start()
        
    def update_label(self, text):
        self.label.setText(text)
    
    #0. Check internet connection
    @QtCore.pyqtSlot(bool)
    def internet_checked(self, internet_access):
        if internet_access:
            # we have internet connectino
            self.update_label('Internet connection confirmed')
        else:
            # No connection available, so tests will fail
            self.update_label('No internet available - can not validate version nor track errors and performance ')
            
            # Stop the other tests
            
    @QtCore.pyqtSlot(str)
    def current_version_checked(self, version):
        self.current_version = version
        self.update_label(f'Current version = {version}')
        
    @QtCore.pyqtSlot(str)
    def latest_version_checked(self, version):
        self.latest_version = version
        self.update_label(f'Latest version = {version}')
    
    @QtCore.pyqtSlot(bool, bool, bool)
    def versions_compared(self, version_up_to_date, version_behind, version_ahead):
        if version_up_to_date:
            text = 'Version is up to date, '
        elif version_behind:
            text = 'ATTENTION: Newer version available on Git release, '
        else:
            text = 'Current version is ahead, '
            
        # Add version info
        text += f'current: {self.current_version} - latest: {self.latest_version}'
        
        self.update_label(text)
        
    @QtCore.pyqtSlot(int)
    def update_progressbar(self, val):
        self.progressbar.setValue(val)
        

class SetupChecker(QtCore.QObject):
    REPO_OWNER = 'Erikd1997'    
    REPO_NAME = 'LIVA_processor'
    
    progress = QtCore.pyqtSignal(int)
    
    started = QtCore.pyqtSignal()
    
    internCheckStarted = QtCore.pyqtSignal()
    internCheckFinished = QtCore.pyqtSignal(bool)
    
    getCurrentVersionStarted = QtCore.pyqtSignal()
    getCurrentVersionFinished = QtCore.pyqtSignal(str)
    
    initSentryStarted = QtCore.pyqtSignal()
    initSentryFinished = QtCore.pyqtSignal()
    
    getLatestVersionStarted = QtCore.pyqtSignal()
    getLatestVersionFinished = QtCore.pyqtSignal(str)
    
    checkVersionsStarted = QtCore.pyqtSignal()
    checkVersionsFinished = QtCore.pyqtSignal(bool, bool, bool)
    
    def start(self):
        threading.Thread(
            target=self._execute, daemon=True
        ).start()   

    def _execute(self):
        self.started.emit()
        self.updateProgress(1)
        
        # Wait for 1 second so that the user sees the dialog
        time.sleep(1)
        
        #0. Check internet connection
        self.internCheckStarted.emit()
        time.sleep(0.2)
        internet_access = have_internet()
        self.internCheckFinished.emit(internet_access)
        self.updateProgress(10)
        time.sleep(0.5)
        
        #1. Get current version number
        self.getCurrentVersionStarted.emit()
        time.sleep(0.2)
        version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VERSION')
        with open(version_file, 'r') as f:
            current_version = f.read()
        self.getCurrentVersionFinished.emit(current_version)

        self.updateProgress(40)
        time.sleep(0.5)
        
        #2. Initialise sdk_int
        self.initSentryStarted.emit()
        time.sleep(0.2)
        sentry_sdk.init(
            dsn="https://be87be4f296d5eb035f1159ca396eaf8@o564965.ingest.us.sentry.io/4507822402830336",
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for tracing.
            traces_sample_rate=1.0,
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            # We recommend adjusting this value in production.
            _experiments={
                "profiles_sample_rate": 1.0,  
            },
            release=f"liva_processor@{current_version}"
        )
        self.initSentryFinished.emit()
        
        self.updateProgress(70)
        time.sleep(0.5)
        
        #3. Get version from git
        self.getLatestVersionStarted.emit()
        time.sleep(0.2)
        latest_version = get_git_version(self.REPO_OWNER, self.REPO_NAME)
        self.getLatestVersionFinished.emit(latest_version)
        
        self.updateProgress(90)
        time.sleep(0.5)
        
        #4. Check version from
        self.checkVersionsStarted.emit()
        time.sleep(0.2)
        if Version(latest_version) > Version(current_version):
            # A new release is available
            version_up_to_date = False
            version_behind = True
            version_ahead = False
        elif Version(latest_version) < Version(current_version):
            # For someone reason, the current version is ahead
            version_up_to_date = False
            version_behind = False
            version_ahead = True
        else:
            version_up_to_date = True
            version_behind = False
            version_ahead = False
        self.checkVersionsFinished.emit(version_up_to_date,
                                        version_behind,
                                        version_ahead
                                        )
        
        self.updateProgress(100)
        
    def updateProgress(self, val):
        self.progress.emit(val)


class StartUpDialog(QtWidgets.QMainWindow):
    okPressed = QtCore.pyqtSignal()
    cancelPressed = QtCore.pyqtSignal()
    applyPressed = QtCore.pyqtSignal()
    def __init__(self, parent = None, relative_size = (0.6, 0.6), name = None):
        super(StartUpDialog, self).__init__(parent)
        
        if not isinstance(parent, type(None)):
            size = (int(relative_size[0] * self.parentWidget().width()),
                    int(relative_size[1] * self.parentWidget().height()))
        else:
            size = (400, 400)

if __name__ == '__main__':        
    repo_owner='Erikd1997'    
    repo_name='LIVA_processor'    
    
    print(get_git_version(repo_owner, repo_name))
    
    version_file  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VERSION')
    with open(version_file, 'r') as f:
        version = f.read()
    print(version)