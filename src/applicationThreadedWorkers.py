# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:07:15 2024

Threaded workers used in main widget

@author: ErikD
"""

import sys
from pathlib import Path
import logging
import vtk
import random
import re

from skspatial.objects import Plane, Points

from vtk.util import numpy_support

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets, QtGui


import pivDataAssimilationSurfaceParticlesFraction_SkinFriction as pivDA

class MainSolverWorker(QtCore.QThread):
    finishedGeneral = QtCore.pyqtSignal()
    finishedSetup = QtCore.pyqtSignal()
    finishedGroundPlaneFit = QtCore.pyqtSignal(object)
    finishedMainRun = QtCore.pyqtSignal(object)
    finishedNothing = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    logSignal = QtCore.pyqtSignal(str)
    _availableMethodNumbers = ('0', '1', '2', '3', '4', 'GT')
    
    # def __init__(self, CSSA, methodNumber):
    def __init__(self, RADIUS, methodNumber, useGroundPlaneFit=True, coinInfo={}):
        super(MainSolverWorker, self).__init__()
        
        # Check if methodNumber is correct
        if (not (methodNumber in self._availableMethodNumbers)) and (not (methodNumber in (int(n) for n in self._availableMethodNumbers))):
            raise ValueError(f'Method number {methodNumber} is not one of the '
                             f'available options: {self._availableMethodNumbers}')
        
        # Determine the correct settings depending on the method number
        if methodNumber != 'GT':
            useGridData = int(methodNumber) < 2
            useObjectInfo = int(methodNumber) > 0
            useLSRConstrained = int(methodNumber) >= 4
            VOItype='s'
            coinHeight = None
            coinOverlap = None
            coinFitMethod = None
        else:
            useGridData = False
            useObjectInfo = coinInfo['coinFitConstrained']
            useLSRConstrained = False
            VOItype='c'
            coinHeight = coinInfo['coinHeight']
            coinOverlap = coinInfo['coinOverlap']
            coinFitMethod = coinInfo['coinFitMethod']
            
            
            
        # Initialise a CSSA object
        self.CSSA = pivDA.CompactSphereSupportedApproximation_VTK(RADIUS, 2,
                                                                  volume_mode=False,
                                                                  griddata_mode=useGridData,
                                                                  LSR_constrained=useLSRConstrained,
                                                                  updateGroundEstimate=useGroundPlaneFit,
                                                                  useObjectInfo=useObjectInfo,
                                                                  velMethod=methodNumber,
                                                                  coinHeight=coinHeight,
                                                                  coinOverlap=coinOverlap,
                                                                  fitMethod=coinFitMethod,
                                                                  VOItype=VOItype
                                                                  )
        
    def executeSetup(self, outputmesh, velfield, velfieldCropDim, useBins,
                     modelPD=None, modelPDFilePath=None,
                     coorTracers=None, coorTracersCropDim = (None, None),
                     useSlice = False, usePlaneSlice = False, rSliceSphere = 100.,
                     grid_size = None):
        # Set the flags to let the self.run() function know what to do
        self.flagRunSetup = True
        self.flagFitGroundPlane = False
        self.flagRunMain = False
        
        self.useBins = useBins
        # CSSA.AddOutputMesh_VTK()
        self.outputmeshToAdd = outputmesh
        
        # CSSA.AddModelPolyData_VTK()
        modelPDAndFilePathNotSet = (isinstance(modelPD, type(None)) and 
                                    isinstance(modelPDFilePath, type(None))
                                    )
        if self.CSSA.updateGroundEstimate and modelPDAndFilePathNotSet:
            raise ValueError('No modelPD has been added, whereas the '
                             'method has been invoked with '
                             f'{self.CSSA.updateGroundEstimate} and this '
                             'requires the addition of a modelPD')
        elif modelPDAndFilePathNotSet:
            self.addModelPD = False
        else:
            self.addModelPD = True
            self.modelPDToAdd = modelPD
            if not isinstance(modelPDFilePath, str):
                raise ValueError(f'Does not accept type {type(modelPDFilePath)} '
                                 'for modelPDFilePath. Only type(str) accepted')
            self.modelPDFilePathToAdd = modelPDFilePath
        
        # CSSA.AddGroundPlaneTracers()
        if self.CSSA.updateGroundEstimate and isinstance(coorTracers, type(None)):
            raise ValueError('No coorTracers have been added, whereas the '
                             'method has been invoked with '
                             f'{self.CSSA.updateGroundEstimate} and this '
                             'requires the addition of coorTracers')
        elif isinstance(coorTracers, type(None)):
            self.addGroundPlaneTracers = False
        else:
            self.addGroundPlaneTracers = True
            self.groundPlaneTracersToAdd = coorTracers
            if isinstance(coorTracersCropDim[0], type(None)) and isinstance(coorTracersCropDim[1], type(None)):
                raise ValueError('Enter cropping dimension values for coorTracersCropDim')
            self.groundPlaneTracersDimToCrop = coorTracersCropDim
            
        # CSSA.AddVelocityField_Numpy()
        self.velfieldToAdd = velfield
        self.velfieldDimToCrop = velfieldCropDim
        self.grid_size = grid_size
        
        # Add slicer settings
        self.useSlice = useSlice
        self.usePlaneSlice = usePlaneSlice
        self.rSliceSphere = rSliceSphere
        
        # Run the setup in a thread
        self.start()
        
    def runSetup(self):
        # Emit info logSignal
        self.logSignal.emit('Adding OutputMesh...')
        # CSSA.AddOutputMesh_VTK()
        self.CSSA.AddOutputMesh_VTK(self.outputmeshToAdd)
        delattr(self, 'outputmeshToAdd')
        
        # CSSA.AddModelPolyData_VTK()
        if self.addModelPD:
            # Emit info logSignal
            self.logSignal.emit('Adding (UpSampled) Model PolyData...')
            # Add model polydata
            self.CSSA.AddModelPolyData_VTK(self.modelPDToAdd)
            delattr(self, 'modelPDToAdd')
        
        # CSSA.AddGroundPlaneTracers()
        if self.addGroundPlaneTracers:
            # Emit info logSignal
            self.logSignal.emit('Adding cropped tracers for ground plane estimate...')
            ### Crop using band
            # Determine inner limit
            maskGroundPlaneTracersInnerEdge = pivDA.maskWithOffsetToSTL(self.groundPlaneTracersToAdd.to_numpy()[:,:3],
                                                                        self.modelPDFilePathToAdd,
                                                                        offset = np.abs(self.groundPlaneTracersDimToCrop[0])
                                                                        )
            
            # This function removes tracers inside the ground + model
            maskGroundPlaneTracersOuterEdge = pivDA.mask_by_offset_to_stl(self.groundPlaneTracersToAdd.to_numpy()[:,:3],
                                                                          self.modelPDFilePathToAdd,
                                                                          offset = self.groundPlaneTracersDimToCrop[1]
                                                                          )
            
            maskGroundPlaneTracers = (maskGroundPlaneTracersInnerEdge == 1) & (maskGroundPlaneTracersOuterEdge==0)
            groundPlaneTracersBanded = self.groundPlaneTracersToAdd.to_numpy()[maskGroundPlaneTracers, :3]
            # Add the ground plane tracers
            self.CSSA.AddGroundPlaneTracers(groundPlaneTracersBanded)
            delattr(self, 'groundPlaneTracersToAdd')
        
        
        # Emit info logSignal
        self.logSignal.emit('Adding cropped velocity field...')

        # Crop using band
        # Define the banded velocity field
        maskVelfieldInnerEdge = pivDA.maskWithOffsetToSTL(self.velfieldToAdd.to_numpy()[:,:3],
                                                          self.modelPDFilePathToAdd,
                                                          offset = np.abs(self.velfieldDimToCrop[0])
                                                          )

        maskVelfieldOuterEdge = pivDA.mask_by_offset_to_stl(self.velfieldToAdd.to_numpy()[:,:3],
                                                            self.modelPDFilePathToAdd,
                                                            offset = self.velfieldDimToCrop[1]
                                                            )

        # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
        _, _, maskVelfieldInsideSTL = pivDA.maskCoordinatesInsideVTKPD(self.velfieldToAdd.to_numpy()[:,:3],
                                                                       self.CSSA.modelPD_vtk)
        
        if self.useBins:
            # Mask the binned velocity field
            maskVelfield = ((maskVelfieldInnerEdge == 1) &
                            (maskVelfieldOuterEdge == 0) &
                            (maskVelfieldInsideSTL == 0) &
                            (self.velfieldToAdd['isValid'])
                            )
        else:
            # Mask the binned velocity field
            maskVelfield = ((maskVelfieldInnerEdge == 1) &
                            (maskVelfieldOuterEdge == 0) &
                            (maskVelfieldInsideSTL == 0)
                            )
            
        bandedVelfield = self.velfieldToAdd.to_numpy()[maskVelfield, :]
        
        ## Determine concentration of tracer particles
        # Set the bounds in which the tracer particles are calculated
        if (self.CSSA.velMethod == '2'):
            concentration = 1/self.grid_size**2
            self.CSSA.AddConcentration(concentration)
            self.logSignal.emit(f'Concentration estimated to be {round(concentration, 2)} part/mm^2')
            
            
        elif (self.CSSA.velMethod == '3'):
            xmin = -150; xmax = 150
            ymin = -150; ymax = 150
            zmin = 0; zmax = 150
            
            # Extract all points inside the domain
            mask_p_in_domain = ((self.velfieldToAdd.to_numpy()[:,0] >= xmin) & (self.velfieldToAdd.to_numpy()[:,0] <= xmax) &
                                (self.velfieldToAdd.to_numpy()[:,1] >= ymin) & (self.velfieldToAdd.to_numpy()[:,1] <= ymax) &
                                (self.velfieldToAdd.to_numpy()[:,2] >= zmin) & (self.velfieldToAdd.to_numpy()[:,2] <= zmax) &
                                (maskVelfieldInsideSTL == 0)
                                )
            NumberOfParticles = np.sum(mask_p_in_domain)
            
            # Define filter to obtain domain volume
            volumeSTLFilter = vtk.vtkMassProperties()
            volumeSTLFilter.SetInputData(self.CSSA.modelPD_vtk)
            volumeSTLFilter.Update()
    
            volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) - volumeSTLFilter.GetVolume()
            ratio = volume / NumberOfParticles
            mean_distance = (3/(4*np.pi*ratio))**(1/3)
            concentration = 1 / mean_distance**2
            # concentration = volume / NumberOfParticles
            self.CSSA.AddConcentration(concentration)
            self.logSignal.emit(f'Concentration estimated to be {round(concentration, 2)} part/mm^2')
        
        # Add the velocity field
        self.CSSA.AddVelocityField_Numpy(bandedVelfield, columns=self.velfieldToAdd.columns)
        delattr(self, 'velfieldToAdd')
        
        # Emit info logSignal
        self.logSignal.emit('Building FluidMesh from velocity field...')
        # CSSA.BuildFluidMeshFromVelocityField()
        self.CSSA.BuildFluidMeshFromVelocityField()
        
        # Emit info logSignal
        self.logSignal.emit('Building point locators...')
        # CSSA.BuildPointLocators()
        self.CSSA.BuildPointLocators()
        
        if self.useSlice:
            if self.usePlaneSlice:
                self.CSSA.AddSlicer('plane')
            else:
                self.CSSA.AddSlicer('sphere', r=self.rSliceSphere)
        
        # Compoute normals
        self.normals = pivDA.pointnormals_from_unstructured_vtk(self.CSSA.outputmesh_vtk)
    
    def executeGroundPlaneFit(self, planeFitFolder):
        # Set the flags to let the self.run() function know what to do
        self.flagRunSetup = False
        self.flagFitGroundPlane = True
        self.flagRunMain = False
        
        # Store ground plane fit folder
        self.planeFitFolder = planeFitFolder
        
        # CSSA.ComputeGroundPlaneFit()
        self.start()
    
    def fitGroundPlane(self):
        # Execute ground plane fit
        # Rvary = 8
        self.CSSA.ComputeGroundPlaneFit(dz_ground=5, Rvary=12 , Rlocal=None, minDistanceToObject=5,
                                        track_percentage=True, partitions=100,
                                        recompute=False, savefolder=self.planeFitFolder,
                                        trackProgressStream=self.progress.emit)
        
        # Get information on the ground plane fit
        ## Recompute the fit since it is not saved:
        shiftedPoints = self.CSSA.shiftedPoints[self.CSSA.shiftComputed, :]
        # Change the ground points into a scikit-spatial Points instance
        shiftedPointsSKSpatial = Points(shiftedPoints)
        
        # Compute the best fit for a ground plane
        groundPlaneFit = Plane.best_fit(shiftedPointsSKSpatial )
        
        # Extract the plane components a, b, c and compute d
        planeFitNormal = groundPlaneFit.normal
        a, b, c = groundPlaneFit.normal
        xPLaneCentroid, yPLaneCentroid, zPLaneCentroid = groundPlaneFit.point
        d = -a*xPLaneCentroid - b*yPLaneCentroid - c*zPLaneCentroid
        
        # Save the normal vector, the centroid and the value d for the plane function
        out = (planeFitNormal, groundPlaneFit.point, d)
        
        return out
        
    def executeMainRun(self):
        # Set the flags to let the self.run() function know what to do
        self.flagRunSetup = False
        self.flagFitGroundPlane = False
        self.flagRunMain = True
        
        # CSSA.Run()
        self.start()
        pass
    
    def runMain(self):
        # Execute main run
        self.CSSA.Run(track_percentage=True, partitions=100, c_start=0,
                      dz_ground=5, maxNStepsAlongNormal=100, test=False,
                      findBL=False, trackProgressStream = self.progress.emit)
        
        ### Store the data into the array with:
        # 0: x-coordinate
        # 1: y-coordinate
        # 2: z-coordinate
        # 3: normal x-coordinate
        # 4: normal y-coordinate
        # 5: normal z-coordinate
        # 6: interpolated u-velocity
        # 7: interpolated v-velocity
        # 8: interpolated w-velocity
        # 9: interpolated absolute velocity
        # 10: Wall-Normal Gradient Velocity In-Plane X-component
        # 11: Wall-Normal Gradient Velocity In-Plane Y-component
        # 12: Wall-Normal Gradient Velocity In-Plane Z-component
        # 13: Wall-Normal Gradient Velocity In-Plane Magnitude
        # 14: isValid
        # 15: Local wall-Normal ground shift vector
        # 16: Fitted wall-normal ground shift
        # 17: Ground shift is computed
        # 18: Thresholded Boundary Layer Height
        # 19: Boundary layer is valid
        # 20: # of fluid mesh nodes
        # 21: # of object mesh nodes
        
        # Initialise results dataset
        output = np.zeros((self.CSSA.Ncells,22))
        
        # Assign the normals to the data array
        output[:,3:6] = self.normals
        
        # Assign the coordinates to the data array
        output[:,:3] = self.CSSA.xyz
    
        # Save velocities
        output[:,6] = self.CSSA.u
        output[:,7] = self.CSSA.v
        output[:,8] = self.CSSA.w
        
        # Compute velocity magnitude in output mesh
        output[:,9] = np.linalg.norm(output[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output[:,10:13] = self.CSSA.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output[:,13] = np.linalg.norm(self.CSSA.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output[:,14] = self.CSSA.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output[:,15] = self.CSSA.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output[:,16] = self.CSSA.shiftedPoints[:,2] - self.CSSA.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output[:,17] = self.CSSA.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output[:,18] = self.CSSA.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output[:,19] = self.CSSA.BLvalid.astype(int)
        
        # Save the number of fluid and object mesh nodes
        output[:,20] = self.CSSA.info[:,0]
        output[:,21] = self.CSSA.info[:,1]
                
        # Swap all points where data is NaN with explicit "0"
        output[~self.CSSA.isValid,6:-1] = np.zeros_like(output[0,6:-1])
        
        return output
        
    
    def run(self):
        # Check if setup must be run
        if self.flagRunSetup:
            self.runSetup()
            self.finishedSetup.emit()
        # Check if ground plane fit must be run
        elif self.flagFitGroundPlane:
            groundPlaneFit = self.fitGroundPlane()
            self.finishedGroundPlaneFit.emit(groundPlaneFit)
        # Check if main run must be executed
        elif self.flagRunMain:
            result = self.runMain()
            self.finishedMainRun.emit(result)
        else:
            self.finishedNothing.emit()
        
        # 
        # self.finishedGeneral.emit()
    
# Think about if this would be needed
class UpdateFluidDataWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    logSignal = QtCore.pyqtSignal(str)
    
    def __init__(self):
        super(UpdateFluidDataWorker, self).__init__()
    
    # def execute(self, )

class LoadFluidDataWorker(QtCore.QThread):
    finishedBin = QtCore.pyqtSignal(pd.DataFrame)
    finishedTracers = QtCore.pyqtSignal(pd.DataFrame)
    finished = QtCore.pyqtSignal()
    headerDataSignal = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
    logSignal = QtCore.pyqtSignal(str)
    
    def __init__(self):
        super(LoadFluidDataWorker, self).__init__()
        
    def execute(self, filename, useBins):
        self.useBins = useBins
        self.fluidDataFileName = filename
        self.start()
        
    def actionLoadFluidData(self):
        # Load velocity data depending on the checked button
        if not self.useBins:
            # Add logging
            self.logSignal.emit('Track-based data selected')
            
            # blocksize = 1024  # tune this for performance/granularity
            # update progress bar
            if Path(self.fluidDataFileName).suffix == '.npy':
                self.progress.emit(10)
                
                velfield_tracks = np.load(self.fluidDataFileName, allow_pickle = True)
                self.progress.emit(70)
                
                if np.ma.size(velfield_tracks[0], axis=1) == 8:
                    columns = ['X [mm]', 'Y [mm]', 'Z [mm]', 'U [m/s]', 'V [m/s]',
                               'W [m/s]', 'Timestep [-]', 'Track ID [-]'
                               ]
                else:
                    columns = ['X [mm]', 'Y [mm]', 'Z [mm]', 'U [m/s]', 'V [m/s]',
                               'W [m/s]', 'Timestep [-]', 'Track ID [-]',
                               'Number of Cameras [-]']
                
                fluidDataVelfield = pd.DataFrame(np.vstack(velfield_tracks),
                                                 index=None,
                                                 columns=columns
                                                 )
                self.progress.emit(100)
            else:
                # We read the file by timestep
                with open(self.fluidDataFileName, 'r') as f:
                    for i, row in enumerate(f):
                        pivDA.ScanLine(row)
            
        else:
            # Add logging
            self.logSignal.emit('Bin-based data selected')
            
            # 1. Quickly read out main data from file
            (fileLength, variables,
             generalInfo, zoneInfo) = pivDA.fileQuickScan(self.fluidDataFileName,
                                                          variablesSeparator=', ')
            numberOfRowsToSkip = zoneInfo['startRowData'].iloc[0]
            headerData = {'NDataLines': fileLength,
                          'NHeaderLines': numberOfRowsToSkip ,
                          'Title': generalInfo['Title'],
                          'Variables': variables,
                          'zoneInfo': zoneInfo}
            
            self.headerDataSignal.emit(headerData)
            
            numberOfVariables = len(variables)
            self.logSignal.emit(f'Reading {Path(self.fluidDataFileName).name} with {fileLength:,} lines')
            self.logSignal.emit(f'Skipping {numberOfRowsToSkip} rows of header text')
            
            
            # 2. Define a chunksize of 1/100 of fileLength
            chunksize = int(np.ceil(fileLength/100))
            
            # 3. Initiate a blank dataframe
            fluidDataVelfield = pd.DataFrame()
            
            # 4. Divide the file into chunks and load chunk by chunk
            columnIDs = [0, 1, 2, 3, 4, 5, 20, 21, 22, 51]
            columnHeaders = ["x", "y", "z", "u", "v", "w", "omegax", "omegay", "omegaz", "isValid"]
            self.logSignal.emit(f'Loading {len(columnIDs)}/{numberOfVariables} columns of data:')
            self.logSignal.emit(f'{[variable for i, variable in enumerate(variables) if i in columnIDs]}')
            for i, chunk in enumerate(pivDA.load_csv_velocity(self.fluidDataFileName, names=columnHeaders,
                                                              usecols=columnIDs,
                                                              chunksize=chunksize,
                                                              low_memory=False)):
                # append it to df
                fluidDataVelfield = pd.concat([fluidDataVelfield, chunk],
                                              ignore_index=True)
                
                # update progress bar
                self.progress.emit(i+1)
            
        return fluidDataVelfield

    def run(self):
        # Execute mainFunc
        loadedData = self.actionLoadFluidData()
        
        if self.useBins:
            self.finishedBin.emit(loadedData)
        else:
            self.finishedTracers.emit(loadedData)
            
        self.finished.emit()
        
        
# =============================================================================
# ####################### ARCHIVED CODE
# =============================================================================
    
# def _checkOutputMesh(self, CSSAobject):
#     return hasattr(CSSAobject, 'outputmesh_vtk')

# def _checkVelocityField(self, CSSAobject, fieldtype):
#     return hasattr(CSSAobject, 'fluidmesh_vtk')

# def _checkModelPD(self, CSSAobject):
#     return hasattr(CSSAobject, 'modelPD_vtkUpSampled')

# # Check that all of the required properties have been set.
# if methodNumber == '0':
#     # Check if outputMesh is set
#     checkOutMesh = self._checkOutputMesh(CSSA)
#     # Check if Velocity field is set and binned
#     checkVelField = self._checkVelocityField(CSSA, fieldType='bins')
    
#     if (not checkOutMesh) or (not checkVelField):
#         checksFailedList = []
#         if (not checkOutMesh):
#             checksFailedList += ['OutputMesh not set']
#         if (not checkVelField):
#             checksFailedList += ['Velfield not set correctly']
#         checksFailedListJoined = checksFailedList.join("\n")
#         raise RuntimeError('Not all checks were passed:\n'
#                            f'{checksFailedListJoined}'
#                            )
    
# elif methodNumber == '1':
#     # Check if outputMesh is set
#     checkOutMesh = self._checkOutputMesh(CSSA)
#     # Check if Velocity field is set and binned
#     checkVelField = self._checkVelocityField(CSSA, fieldType='bins')
#     # Check if model STL is set
#     checkModelPD = self._checkModelPD(CSSA)
    
#     if (not checkOutMesh) or (not checkVelField) or (not checkModelPD):
#         checksFailedList = []
#         if (not checkOutMesh):
#             checksFailedList += ['OutputMesh not set']
#         if (not checkVelField):
#             checksFailedList += ['Velfield not set correctly']
#         checksFailedListJoined = checksFailedList.join("\n")
#         raise RuntimeError('Not all checks were passed:\n'
#                            '{checksFailedListJoined}'
#                            )
    
# elif methodNumber == '2':
#     # Check if outputMesh is set
#     checkOutMesh = self._checkOutputMesh(CSSA)
#     # Check if Velocity field is set and binned
#     checkVelField = self._checkVelocityField(CSSA, fieldType='bins')
#     # Check if model STL is set
#     checkModelPD = self._checkModelPD(CSSA)
    
#     if (not checkOutMesh) or (not checkVelField) or (not checkModelPD):
#         checksFailedList = []
#         if (not checkOutMesh):
#             checksFailedList += ['OutputMesh not set']
#         if (not checkVelField):
#             checksFailedList += ['Velfield not set correctly']
#         checksFailedListJoined = checksFailedList.join("\n")
#         raise RuntimeError('Not all checks were passed:\n'
#                            f'{checksFailedListJoined}'
#                            )
    
# elif methodNumber == '3':
#     # Check if outputMesh is set
#     checkOutMesh = self._checkOutputMesh(CSSA)
#     # Check if Velocity field is set and binned
#     checkVelField = self._checkVelocityField(CSSA, fieldType='tracers')
#     # Check if model STL is set
#     checkModelPD = self._checkModelPD(CSSA)
    
#     if (not checkOutMesh) or (not checkVelField) or (not checkModelPD):
#         checksFailedList = []
#         if (not checkOutMesh):
#             checksFailedList += ['OutputMesh not set']
#         if (not checkVelField):
#             checksFailedList += ['Velfield not set correctly']
#         checksFailedListJoined = checksFailedList.join("\n")
#         raise RuntimeError('Not all checks were passed:\n'
#                            f'{checksFailedListJoined}'
#                            )
    
# elif methodNumber == '4':
#     # Check if outputMesh is set
#     checkOutMesh = self._checkOutputMesh(CSSA)
#     # Check if Velocity field is set and binned
#     checkVelField = self._checkVelocityField(CSSA, fieldType='tracers')
#     # Check if model STL is set
#     checkModelPD = self._checkModelPD(CSSA)
    
#     if (not checkOutMesh) or (not checkVelField) or (not checkModelPD):
#         checksFailedList = []
#         if (not checkOutMesh):
#             checksFailedList += ['OutputMesh not set']
#         if (not checkVelField):
#             checksFailedList += ['Velfield not set correctly']
#         checksFailedListJoined = checksFailedList.join("\n")
#         raise RuntimeError('Not all checks were passed:\n'
#                            f'{checksFailedListJoined}'
#                            )
# else:
#     raise ValueError(f'Method number {methodNumber} is not one of the '
#                      f'available options: {self._availableMethodNumbers}')

# self.CSSA = CSSA