# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:43:29 2024

Script to evaluate the near-surface fluid-dynamics with updating for
ground-estimate.

@author: ErikD
"""

# This script contains functions to input several grid files and output+save a vtk file
import os
import re
import logging
from pathlib import Path
# from operator import itemgetter
import time
import warnings
import functools
from itertools import islice
# import math

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

import numba
import vtk
import pyvista as pv
from vtk.util import numpy_support
from vtkbool.vtkBool import vtkPolyDataBooleanFilter
from vtk.numpy_interface import dataset_adapter as dsa
import open3d as o3d

# from copy import deepcopy

from skspatial.objects import Plane, Points
from collections.abc import Iterator

# Import own modules
import optimizeFuncPIV as pivOpt

def GetRodriguesTransformationMatrix(source, destination, threshold=1e-5):
    if np.linalg.norm(source - destination) < threshold:
        return np.eye(4)
    
    # Compute rotation axis and rotation angle
    rotationAxis = np.cross(source, destination)
    rotationAxisNormalized = rotationAxis / np.linalg.norm(rotationAxis)
    rotationAngle = np.arccos(np.dot(source, destination))
    return GRTN(rotationAxisNormalized, rotationAngle)
    
def GRTN(rotationAxisNormalized, rotationAngle):
    # Set up the K matrix
    kx, ky, kz = rotationAxisNormalized
    Kmatrix = np.array([[0., -kz, ky],
                        [kz, 0., -kx],
                        [-ky, kx, 0.]])
    
    # Compute Rodrigues' rotation
    rotationMatrix = np.identity(3) + np.sin(rotationAngle) * Kmatrix + (1-np.cos(rotationAngle)) * (Kmatrix @ Kmatrix)
    
    # Convert to a 4x4 transformation matrix
    transformMatrix = np.zeros((4,4))
    transformMatrix[:3,:3] = rotationMatrix
    transformMatrix[3,3] = 1.
    return transformMatrix

def CreateVTKPolyDataFromTrianglePointsNP(obj):
    vtkpd = vtk.vtkPolyData()
    vtkpdCells = vtk.vtkCellArray()
    vtkpdPoints = vtk.vtkPoints()
    
    vtkpdCells.InsertNextCell(len(obj))
    
    newPointId = 0
    # Add each point to the vtkPoints instance
    for idx, point in enumerate(obj):
        vtkpdPoints.InsertPoint(newPointId,
                                point[0],
                                point[1],
                                point[2])
        # Add pointID to the cell array
        vtkpdCells.InsertCellPoint(newPointId)
        # Increment pointID
        newPointId += 1
    
    vtkpd.SetPoints(vtkpdPoints)
    vtkpd.SetPolys(vtkpdCells)
    return vtkpd
    
def CreateVTKPolyDataAsCircle(radius, center):
    spherepd = vtk.vtkSphereSource()
    spherepd.SetCenter(center)
    spherepd.SetRadius(radius)
    spherepd.Update()
    return spherepd.GetOutput()

# Try to sprinkle in some use of numba to speed up the execution
@numba.njit(numba.types.UniTuple(numba.float32,2)(numba.float32, numba.float32, numba.float32, numba.float32))
def numba_fit_quad_polynomial_with_zero(x1, y1, x2, y2):
    a2 = (y2 * x1 - y1 * x2) / (x2**2*x1 - x1**2 * x2)
    a1 = (y1 - a2*x1**2) / x1
    return a1, a2

def convVTKtoOpen3DTriangleMesh_FilesLegacy(polydata, tmp_folder=r'C:\temp'):
    t0 = time.time()
    # Create a temporary STL file to store the vtkCleanPolyData() 
    # to load it in with o3d again
    tmp_file = tmp_folder + r'\PolyDataCutSphere-Test.stl'
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(tmp_file)
    writer.SetInputData(polydata)
    writer.Write()
    
    # Use open3d to load in the file
    mesh = o3d.io.read_triangle_mesh(tmp_file)
    
    os.remove(tmp_file)
    t1 = time.time()
    
    return mesh

def convVTKtoOpen3DTriangleMesh(trianglesPD):
    # The input will definitely be a triangle mesh, hence there is no need to
    # apply a triangulation filter input

    # 1. Create point array
    points = numpy_support.vtk_to_numpy(trianglesPD.GetPoints().GetData())
    
    # 2. Create triangle connectivity array
    triangleCells = trianglesPD.GetPolys()
    connectivity = numpy_support.vtk_to_numpy(triangleCells.GetConnectivityArray()).astype(np.int32)
    
    triangleIndices = np.empty((triangleCells.GetNumberOfOffsets()-1, 3))
    for i in range(triangleCells.GetNumberOfOffsets()-1):
        triangleIndices[i,:] = connectivity[triangleCells.GetOffset(i):triangleCells.GetOffset(i+1)]
        #[connectivity[triangleCells.GetOffset(i):triangleCells.GetOffset(i+1)] for i in range(triangleCells.GetNumberOfOffsets()-1)]
    
    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangleIndices))
        
    return mesh

def createSphereSurface(center, radius, thetaRes=20, phiRes=20):
    # Create a sphere object
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetCenter(*center)
    sphere.SetThetaResolution(thetaRes)
    sphere.SetPhiResolution(phiRes)
    sphere.Update()
    spherePD = sphere.GetOutput()
    
    # Get outside surface of the sphere
    # Build a cell locator for the sphere
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(spherePD)
    cellLocator.BuildLocator()
    
    # Fire a ray from outside to obtain a first outer cell
    rayStart = center + 1.1*radius
    
    cellID = vtk.reference(-1)
    xyz = np.array([0.,0.,0.])
    t = vtk.reference(-1.)
    pcoords = np.array([0.,0.,0.])
    subId = vtk.reference(-1)
    
    intersectFound = cellLocator.IntersectWithLine(rayStart,
                                                   center,
                                                   0.0001,
                                                   t,
                                                   xyz,
                                                   pcoords,
                                                   subId,
                                                   cellID)
    # print(f'Found intersection {intersectFound==1}. Id of cell on outside surface: {cellID}')
    
    # Use vtkPolDataConnectivityFilter to extract the outer surface
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(spherePD)
    connectivityFilter.SetExtractionModeToCellSeededRegions()
    connectivityFilter.InitializeSeedList()
    connectivityFilter.AddSeed(cellID)
    connectivityFilter.Update()
    sphereSurfacePD = connectivityFilter.GetOutput()
    
    # print(f'Showing results. Number of cells: {sphereSurfacePD.GetNumberOfCells()}')
    
    return sphereSurfacePD

def SamplePointsOnIntersection(objectPD, center, radius,
                               NumberOfVolumePoints, griddata_mode, C,
                               tmp_folder=r'C:\temp',
                               thetaRes=20, phiRes=20, returnIntersect=False):
    # Offset the center a little bit, so it never coincides with cells of the 
    # objectPD, cus then we're in trouble...
    mini_offset = 1e-2
    mini_offset_vector = np.ones(3) * mini_offset
    center += mini_offset_vector
    
    ###############################################
    ###### B. Get the surface of the cut model
    ###############################################
    # 1. Construct the sphere (again but as a function... it is what it is)
    sphere = vtk.vtkSphere()
    sphere.SetRadius(radius)
    sphere.SetCenter(*center)

    # Use clip polydata to clip the model data
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(sphere)
    clipper.SetInputData(objectPD)
    clipper.GenerateClippedOutputOff()
    clipper.GenerateClipScalarsOn()
    clipper.InsideOutOn()
    clipper.Update()
        
    # Create a temporary STL file to store the vtkCleanPolyData() 
    # to load it in with o3d again
    # tmp_file = tmp_folder + r'\PolyDataCutSphere-Test.stl'
    # writer = vtk.vtkSTLWriter()
    # writer.SetFileName(tmp_file)
    # writer.SetInputData(clipper.GetOutput())
    # writer.Write()
    
    # Use open3d to load in the file
    # mesh = o3d.io.read_triangle_mesh(tmp_file)
    mesh = convVTKtoOpen3DTriangleMesh(clipper.GetOutput())
    
    if griddata_mode:
        NumberOfSurfacePoints = 3
    else:
        
        
        # Get the area through vtkMassProperties()
        propertiesSurface = vtk.vtkMassProperties()
        propertiesSurface.SetInputData(clipper.GetOutput())
        propertiesSurface.Update()
        surfaceArea = propertiesSurface.GetSurfaceArea()
        
        ###############################################
        ###### B. Get the volume of intersection
        ###############################################
        
        # # 1. Create a sphereSurface polydata
        # sphereSurfacePD = createSphereSurface(center+mini_offset_vector, radius)
        
        # # 2. Compute volume intersection
        # boolintersect = vtk.vtkBooleanOperationPolyDataFilter()
        # boolintersect.SetOperationToIntersection()
        # boolintersect.SetInputData(0, sphereSurfacePD)
        # boolintersect.SetInputData(1, objectPD)
        # boolintersect.Update()
        # sphereSurfaceCutPD = boolintersect.GetOutput(0)
        
        # # 3. Get volume through vtkMassProperties()
        # propertiesIntersect = vtk.vtkMassProperties()
        # propertiesIntersect.SetInputData(sphereSurfaceCutPD)
        # propertiesIntersect.Update()
        # fluidVolume = 4/3 * np.pi * radius**3 # - propertiesIntersect.GetVolume()
        
        
        NumberOfSurfacePoints = int(round(surfaceArea * C))
        
        # # Get equivalent radii
        # RadiusSurfaceEquivalent = np.sqrt(surfaceArea / np.pi)
        
        # volume = NumberOfVolumePoints / C
        # RadiusVolumeEquivalent = (3/4 * volume / np.pi)**(1/3)
        
        # # RadiusVolumeEquivalent = (3/4 * fluidVolume / np.pi)**(1/3)
        
        # # Compute number of surface points
        # # NumberOfSurfacePoints = round(16.29 * NumberOfVolumePoints**(2/3) * (RadiusSurfaceEquivalent / radius))#RadiusVolumeEquivalent))
        # NumberOfSurfacePoints = int(round(NumberOfVolumePoints**(2/3) * (RadiusSurfaceEquivalent / RadiusVolumeEquivalent)**(2/3)))
    
    # Apply poisson disk sampling
    # Set the seed to control how poisson disk sampling operates
    o3d.utility.random.seed(0)
    pcd = mesh.sample_points_poisson_disk(NumberOfSurfacePoints )
    pcd_points = np.asarray(pcd.points)
    
    # Remove tmp_file
    # os.remove(tmp_file)
    
    if returnIntersect:
        return pcd_points, NumberOfSurfacePoints, clipper.GetOutput()
    else:
        return pcd_points, NumberOfSurfacePoints

def numpy_points_to_vtk_polydata(points):
    # Cast points into vtk character array object
    vtk_chararray = numpy_support.numpy_to_vtk(num_array=points, deep=True, array_type=vtk.VTK_FLOAT)

    # Cast points to check into vtk points object
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_chararray)

    # Cast points to check into vtk polydata object
    vtk_polydata = vtk.vtkPolyData()
    vtk_polydata.SetPoints(vtk_points)

    return vtk_polydata

def maskWithOffsetToSTL(coor_np, stl_filepath, offset=0, offsetGP=True):
    # Ensure that coor_np is of size (N, 3)
    assert np.ma.size(coor_np, axis=1) == 3, 'Input coor_np does not have the correct size. It must be of size (N, 3).'
    assert offset >= 0, 'Only positive offsets are accepted.'
    
    # Find all tracer particles inside the STL
    _, _, stl_mask = enclosed_points_with_stl(coor_np, stl_filepath)
    
    # Use open3d to compute the unsigned distances
    mesh = o3d.io.read_triangle_mesh(stl_filepath)
    unsignedDistances = distance_from_points_to_o3dmesh(mesh, coor_np, output_scene=False)
    
    # Define the mask in two-fold
    #1. Points inside the STL
    isValidInsideSTL = (stl_mask==1) & (unsignedDistances < offset)
    
    #2. Points outside STL and above ground
    if offsetGP:
        isValidOutsideSTL = (stl_mask==0) & (coor_np[:,2] > -offset)
    else:
        isValidOutsideSTL = (stl_mask==0)
    
    isValid = isValidInsideSTL | isValidOutsideSTL
    
    return isValid

def getVTKCellCenter(cell):
    # Function to compute center of a VTK cell
    # 1. Get parametric center of cell
    centerParametric = np.zeros(3)
    cell.GetParametricCenter(centerParametric)
    
    # 2. Get XYZ center of cell
    centerXYZ = np.zeros(3)
    weights = np.zeros(cell.GetNumberOfPoints())
    subID = vtk.reference(-1)
    cell.EvaluateLocation(subID, centerParametric, centerXYZ, weights)
    
    return centerXYZ

def maskCoordinatesInsideVTKPD(points_to_check, modelPD_vtk):
    # Cast points to check into vtk character array object
    points_to_check_vtk_chararray = numpy_support.numpy_to_vtk(num_array=points_to_check,
                                                               deep=True,
                                                               array_type=vtk.VTK_FLOAT)
    
    # Cast points to check into vtk points object
    points_to_check_vtk_points = vtk.vtkPoints()
    points_to_check_vtk_points.SetData(points_to_check_vtk_chararray)
    
    # Cast points to check into vtk polydata object
    points_to_check_vtk_polydata = vtk.vtkPolyData()
    points_to_check_vtk_polydata.SetPoints(points_to_check_vtk_points)

    enclosed_points = vtk.vtkSelectEnclosedPoints()
    enclosed_points.SetSurfaceData(modelPD_vtk)
    enclosed_points.SetInputData(points_to_check_vtk_polydata)
    enclosed_points.Update()
    enclosed_points_vtk_polydata = enclosed_points.GetPolyDataOutput()

    # Extract the enclosed_points data from polydata -> pointdata -> attribute (character array)
    mask = numpy_support.vtk_to_numpy(enclosed_points_vtk_polydata.GetPointData().GetAttribute(0))
    points_outside = points_to_check[mask==0, :]
    points_inside = points_to_check[mask==1, :]
    
    return points_outside, points_inside, mask

def mask_by_offset_to_stl(coor_np, stl_filepath, offset=0, offsetGP=True):
    
    # Ensure that coor_np is of size (N, 3)
    assert np.ma.size(coor_np, axis=1) == 3, 'Input coor_np does not have the correct size. It must be of size (N, 3).'
    assert offset >= 0, 'Only positive offsets are accepted.'
        
    # Remove all points that lie inside off the object
    _, _, stl_mask = enclosed_points_with_stl(coor_np, stl_filepath)
    
    # Use open3d to compute the unsigned distances
    mesh = o3d.io.read_triangle_mesh(stl_filepath)
    unsignedDistances = distance_from_points_to_o3dmesh(mesh, coor_np, output_scene=False)
    
    if offsetGP:
        # Mask based on the unsigned stl_mask, distance from STL and distance from ground
        isValid = (stl_mask==0) & (unsignedDistances > offset) & (coor_np[:,2] > offset)
    else:
        isValid = (stl_mask==0) & (unsignedDistances > offset)
    
    return isValid

class CompactSphereSupportedApproximation_VTK(object):
    vorticityNameXOptions = ['omegax', 'omega x', 'omega_x', 'vorx', 'vor x', 'vor_x',
                             'vorticityx', 'vorticity x', 'vorticity_x',
                             'vorticity w_x', 'vorticity wx', 'vorticityw_x', 'vorticitywx', ]
    vorticityNameYOptions = ['omegay', 'omega y', 'omega_y', 'vory', 'vor y', 'vor_y',
                             'vorticityy', 'vorticity y', 'vorticity_y',
                             'vorticity w_y', 'vorticity wy', 'vorticityw_y', 'vorticitywy', ]
    vorticityNameZOptions = ['omegaz', 'omega z', 'omega_z', 'vorz', 'vor z', 'vor_z',
                             'vorticityz', 'vorticity z', 'vorticity_z',
                             'vorticity w_z', 'vorticity wz', 'vorticityw_z', 'vorticitywz', ]
    
    _VOItypeOptions = ['S', 's', 'C', 'c']
    _VOItypeDescriptions = {'S': 'Spherical',
                            'C': 'Coin-stacked (cylindrical)'}
    
    def __init__(self, radius, val2, volume_mode=False,
                 griddata_mode=False, LSR_constrained=False,
                 updateGroundEstimate=False, useObjectInfo=False,
                 velMethod='0', VOItype='S', coinHeight=None,
                 coinOverlap=None, fitMethod=None):
        # updateGroundEstimate can be used to first update ground estimate
        # over the entirety of the outputmesh and subsequently perform the computations
        
        # Upon initialization it is possible to choose for
        # a VolumeOfInterest (VOI) of 'S'/'s' = Sphere or 'C'/'c' = Cylinder.coin
        
        self.VOItype = VOItype.upper()
        if VOItype.upper() == 'S':
            self.R = radius
            self.LSR_ORDER = val2
            self.volume_mode = volume_mode
            self.griddata_mode = griddata_mode
            self.LSR_constrained = LSR_constrained
            self.useObjectInfo = useObjectInfo
            
            # Define flag which to use
            self.useSphere = True
            self.useCoin = False
        elif VOItype.upper() == 'C':
            assert isinstance(coinHeight, (int, float)), 'Enter a number for coingHeight'
            assert coinOverlap >= 0 and coinOverlap < 100, 'Enter a percentage for coinOverlap between [0 - 100>'
            self.R = radius
            # self.H = vale2
            self.NCoins = val2
            self.coinHeight = coinHeight
            self.coinOverlap = coinOverlap
            self.volume_mode = False
            self.griddata_mode = False
            self.LSR_constrained = False
            self.useObjectInfo = useObjectInfo
            self.coinFittingMethod = fitMethod
            # self.NCoins = coinOverlap / 100 * coinHeight
            
            # Define flag which to use
            self.useSphere = False
            self.useCoin = True
        else:
            
            _VOItypePrintDescriptions = [f"'{tp}' for {self._VOItypeDescriptions[tp.upper()]}" for tp in self._VOItypeOptions]
            _VOItypePrintDescriptionsText = ', or\n'.join(_VOItypePrintDescriptions)
            raise ValueError('Inapproriate choice of VOIType. Avalaible choices are:\n'
                             f"{_VOItypePrintDescriptionsText}")
        
        self.velMethod = velMethod
        self.updateGroundEstimate = updateGroundEstimate
        
        # Initialise groundPlaneFitComputed variable
        self.groundPlaneFitComputed = False
        
        # Initialise slicer function
        self.sliceFunc = lambda x, y, z: np.ones_like(z, dtype=bool)
        self.sliceFuncAdded = False
    
    ###########################################################################
    #### All core methods used by algorithm
    def _get_cell_info(self, c_idx):
        # Get the cell and centroid
        vtk_cell = self.outputmesh_vtk.GetCell(c_idx)
        cell_centroid = getVTKCellCenter(vtk_cell)
        
        # Compute cell normal
        cellNormal = np.array([0., 0., 0.])
        cellPoints = vtk_cell.GetPoints()
        vtk_cell.ComputeNormal(cellPoints.GetPoint(0), cellPoints.GetPoint(1), cellPoints.GetPoint(2), cellNormal)
        
        return vtk_cell, cell_centroid, cellNormal
    
    def _find_ground_shift_for_cell(self, c_idx, vtk_point, vtk_cell,
                                    distanceToObject, funcRotateForward,
                                    funcRotateBackward, Rvary, dz_ground,
                                    Rlocal, minDistanceToObject):
        # The local velocity field is shifted and rotated
        if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
            # Determine the shift in ground plane
            if self.groundPlaneFitComputed:
                shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
            else:
                shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                              Rvary, dz_ground, Rlocal=Rlocal)
            
            shiftVectorLocal = np.zeros(3,)

    def _localize_velfield_and_shift(self,):
        # Use a localized velocity field and shift the velocity field
        velfieldLocal = self._getLocalVelfield(self.pointLocator_fluidmesh, Rlocal, vtk_point, self.velfield)
        coorLocalTranslated = velfieldLocal[:,:3] - shiftVectorLocal
        uvwLocal = velfieldLocal[:,3:6]
        
        # Crop the coordinates if they fall inside the ground
        maskCoordinatesLocal = coorLocalTranslated[:,2] >= 0
        
        coorLocalTransformedCropped = funcRotateForward(coorLocalTranslated[maskCoordinatesLocal, :] - vtk_point)
        uvwLocalCropped = uvwLocal[maskCoordinatesLocal, :]
        
        # Build the translated velocity field
        (fluidmeshLocalTranslated_vtk,
         velfieldLocalTranslated) = self._buildTransformedLocalFluidMesh(coorLocalTransformedCropped,
                                                                         uvwLocalCropped,
                                                                         velfieldLocal[maskCoordinatesLocal,6:]
                                                                         )

        # Update the fluidmesh point locator
        fluidmeshToUse = fluidmeshLocalTranslated_vtk
        velfieldToUse = velfieldLocalTranslated
        self.velfieldToUse = velfieldLocalTranslated
        self.shift = shiftVectorLocal
        self.angle = rotationAngle
        
        # Build a pointLocator for the local fluid mesh
        pointLocator_fluidmeshToUse = self._build_vtkPointLocator(fluidmeshToUse, 100)
    
    ###########################################################################
    #### Remaining methods which are use more generally
    
    def GetRotationInfo(self, rotateSource, rotateDestination, threshold=1e-5):
        if np.linalg.norm(rotateDestination - rotateSource) < 1e-5:
            # If the normal direction is already pointing up, we don't need to rotate anything
            funcRotateForward = lambda vec: vec
            funcRotateBackward = lambda vec: vec
            rotationAngle = 0.
            rotMat = np.eye(4)
        else:
            rotationAxis = np.cross(rotateSource, rotateDestination)
            rotationAxisNormalized = rotationAxis / np.linalg.norm(rotationAxis)
            rotationAngle = np.arccos(np.dot(rotateSource, rotateDestination))
        
            # Use Rodrigues' rotation formula
            funcRotateForward = lambda vec: self._RodriguesRotation(vec, rotationAxisNormalized, rotationAngle)
            funcRotateBackward = lambda vec: self._RodriguesRotation(vec, rotationAxisNormalized, -1*rotationAngle)
            
            rotMat = GRTN(rotationAxisNormalized, rotationAngle)
        return funcRotateForward, funcRotateBackward, rotationAngle, rotMat
    
    def FindClosestPointToObject(self, point):
        # Initialize variables
        closest_point = np.array([0.0, 0.0, 0.0])
        cell_obj = vtk.vtkGenericCell()
        projected_cellId_obj = vtk.reference(0)
        subId_obj = vtk.reference(0)
        dist2_obj = vtk.reference(0.)
        
        # Compute the closest point
        self.cellLocator_objectSTL.FindClosestPoint(point,
                                                    closest_point,
                                                    cell_obj,
                                                    projected_cellId_obj,
                                                    subId_obj,
                                                    dist2_obj
                                                    )
        return closest_point, dist2_obj.get()
    
    def _vtk_idlist_to_lst(self, vtkIdLst):
        N = vtkIdLst.GetNumberOfIds()
        return [vtkIdLst.GetId(idx) for idx in range(N)]
    
    def _find_ids_in_sphere_with_Locator(self, Locator, R, point):
        ids_in_sphere = vtk.vtkIdList()
        Locator.FindPointsWithinRadius(R, point, ids_in_sphere)
        return self._vtk_idlist_to_lst(ids_in_sphere)
        
    def _find_ids_in_column_with_Cylinder(self, fluidmeshVTK, R,
                                          point, columnHeight, normal,
                                          rotMat):
        # Define the column center
        plane1Center = point + normal * columnHeight
        plane2Center = point
        columnCenter = point + columnHeight/2 * normal
        
        # Tag the data with ids
        # ids_array = numpy_support.numpy_to_vtk(range(fluidmeshVTK.GetNumberOfPoints()), array_type=0)
        # ids_array.SetName('ids')
        # fluidmeshVTK.GetPointData().AddArray(ids_array)
        
        # # Use implicit filtering
        # cylinderFilter = vtk.vtkCylinder()
        # cylinderFilter.SetCenter(*columnCenter)
        # cylinderFilter.SetRadius(R)
        # cylinderFilter.SetAxis(*normal)
        
        # # plane
        # planeFilter1 = vtk.vtkPlane()
        # planeFilter1.SetOrigin(plane1Center)
        # planeFilter1.SetNormal(normal)
        
        # planeFilter2 = vtk.vtkPlane()
        # planeFilter2.SetOrigin(plane2Center)
        # planeFilter2.SetNormal(-1*normal)
        
        # # Combine filters
        # booleanFilter = vtk.vtkImplicitBoolean()
        # booleanFilter.AddFunction(planeFilter2)
        # booleanFilter.AddFunction(cylinderFilter)
        # booleanFilter.AddFunction(planeFilter1)
        # booleanFilter.SetOperationTypeToIntersection()
        
        # enclosedPointsColumn = vtk.vtkExtractPoints()
        # enclosedPointsColumn.SetInputData(fluidmeshVTK)
        # enclosedPointsColumn.SetImplicitFunction(booleanFilter)
        # enclosedPointsColumn.Update()
        # self.enclosedPointsColumn = enclosedPointsColumn
        
        # Create the column
        baseColumn = vtk.vtkCylinderSource()
        baseColumn.SetCenter((0, 0, 0))
        baseColumn.SetRadius(R)
        baseColumn.SetHeight(columnHeight)
        baseColumn.SetResolution(100)
        baseColumn.Update()
        self.baseColumn = baseColumn
        
        # # Transform the cylinder to align with the z-axis
        baseColumnTransform = vtk.vtkTransform()
        baseColumnTransform.PostMultiply()
        baseColumnTransform.RotateX(-90.0)
        baseColumnTransform.Concatenate(rotMat.flatten())
        baseColumnTransform.Translate(*columnCenter)
        baseColumnTransform.Update()
        self.baseColumnTransform = baseColumnTransform
        
        column = vtk.vtkTransformPolyDataFilter()
        column.SetTransform(baseColumnTransform)
        column.SetInputData(baseColumn.GetOutput())
        column.Update()
        self.column = column
        
        #b. Find points enclosed inside cylinderVary
        enclosedPoints = vtk.vtkSelectEnclosedPoints()
        enclosedPoints.SetSurfaceData(column.GetOutput())
        enclosedPoints.SetInputData(fluidmeshVTK)
        enclosedPoints.Update()
        enclosedPointsColumn = enclosedPoints.GetPolyDataOutput()
        self.enclosedPointsColumn = enclosedPointsColumn
                    
        #c. Use coinMask to extract all fluid mesh particles inside the cylinder
        columnMask = numpy_support.vtk_to_numpy(enclosedPointsColumn.GetPointData().GetAttribute(0))
        fluidmeshColumnIds = np.where(columnMask)[0]
        # fluidmeshColumnIds = numpy_support.vtk_to_numpy(enclosedPointsColumn.GetOutput().GetPointData().GetArray('ids'))
        
        return fluidmeshColumnIds 
    
    def _build_vtkPointLocator(self, vtk_dataset, NumberOfPointsPerBucket):
        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(vtk_dataset)
        pointLocator.AutomaticOn()
        pointLocator.SetNumberOfPointsPerBucket(NumberOfPointsPerBucket)
        pointLocator.BuildLocator()
        return pointLocator
    
    def _build_vtkCellLocator(self, vtk_dataset):
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(vtk_dataset)
        cellLocator.BuildLocator()
        return cellLocator
    
    def _find_ground_intersect(self, point, R_sphere, z_ground=0):
        # Find the center of the circle
        h_sphere = point[2] - z_ground
        
        # Return None when the sphere does not intersect with the ground
        if h_sphere > R_sphere:
            return None
        
        center_point = np.array(point) - np.array([0, 0, h_sphere])
        
        # Find the radius of the circle
        radius = np.sqrt(R_sphere**2 - h_sphere**2)
        
        return center_point, radius
    
    def _get_vtk_cells_by_pointIds(self, vtk_mesh, p_ids):
        # Initialize the dictionary with the point indices
        out = dict.fromkeys(p_ids)
        
        for p_idx in p_ids:
            # Create a vtkIdList instance
            vtkIdList = vtk.vtkIdList()
            # Get cells connected to the point
            vtk_mesh.GetPointCells(p_idx, vtkIdList)
            # Convert this to a numpy array
            IdList = self._vtk_idlist_to_lst(vtkIdList)
            # Add result to dictionary
            out[p_idx] = IdList
        
        return out
    
    def _slice_circular_array(self, array, slc):
        if slc.stop >= np.ma.size(array, axis=1):
            array = np.hstack((array, array))
        
        return array[slc]
    
    def _X_for_tri_polynomial(self, points, order=2):
        Nrows = np.ma.size(points, axis=0)
        if order == 1:
            Ncols = 4
        elif order == 2:
            Ncols = 10
        elif order == 3:
            raise NotImplementedError(f'Order is {order}. Only order up to and including 2 is implemented.')
        else:
            raise NotImplementedError(f'Order is {order}. Only order up to and including 2 is implemented.')

        # Initialize the matrix
        matrix = np.zeros((Nrows, Ncols))

        # Fill the matrix column-wise
        for idx_col in range(Ncols):
            # Split the filling of the matrix between order 0 (1, 1, 1, etc...) and order N for clarity
            if idx_col == 0:
                matrix[:,idx_col] = 1
            elif idx_col > 0 and idx_col < 7:
                ## The nth power terms
                matrix[:, idx_col] = points[:, (idx_col-1)%3] ** ( 1 + ((idx_col-1)//3))
            else:
                ## The cross terms
                # Define the indices to multiply
                idcs = ((idx_col-1)%3-1, (idx_col-1)%3)
                matrix[:,idx_col] = points[:, idcs[0]] * points[:, idcs[1]]

        return matrix

    def _LSR_Y(self, values):
        return values.reshape((-1,1), order='F')

    def _least_squares_solve(self, X, Y):
        return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

    def _estimate_velocity_at_sphere_center_with_LSR(self, velfield_coor, velfield_values, order=2, returnYfit=False, return_beta=False):        
        X = self._X_for_tri_polynomial(velfield_coor, order=order)
        Y = self._LSR_Y(velfield_values)
        beta = self._least_squares_solve(X, Y)
        
        # Lastly, compute the residuals
        Yfit = np.dot(X, beta)
        residuals = Y - Yfit
        
        if returnYfit:
            return beta[0,0], beta, residuals, Yfit
        else:
            return beta[0,0], beta, residuals
    
    def _estimate_velocity_at_sphere_center_with_LSR_divfree(self, velfield_coor, velfield_values, order=2):
        assert np.ma.size(velfield_values, axis=1) == 3, 'All three components of the velocity field need to be provided.'
        
        # Build the matrix for tri-nonlinear polynomial as before
        X = self._X_for_tri_polynomial(velfield_coor, order=order)
        
        # Determine the number of coefficients per velocity component
        n_of_coefficients = order*3 + 1
        
        # Create the vector holding the velocities u, v and w
        U = self._LSR_Y(velfield_values[:,0])
        V = self._LSR_Y(velfield_values[:,1])
        W = self._LSR_Y(velfield_values[:,2])
        UVW = np.hstack((U, V, W)).reshape((-1,1))
        
        # Create the matrix A which contains the matrix X along its diagonal
        fill_matrix = np.zeros_like(X)
        A_row1 = np.hstack((X,           fill_matrix, fill_matrix))
        A_row2 = np.hstack((fill_matrix, X,           fill_matrix))
        A_row3 = np.hstack((fill_matrix, fill_matrix, X))
        A = np.vstack((A_row1, A_row2, A_row3))
        
        # Create the matrix R which holds the divergence free constraint at the point
        R = np.zeros((1, 3*n_of_coefficients))
        R[0, [1, 2+n_of_coefficients, 3+n_of_coefficients*2]] = 1
        
        # Construct the LHS_matrix and RHS_vector to solve the Lagrangian problem
        LHS_matrix = np.hstack((np.vstack((2*np.dot(A.T,A), R)),
                                np.vstack((R.T, np.zeros((1, 1)) )) ))
        RHS_vector = np.vstack((2*np.dot(A.T,UVW), np.zeros((1 , 1))))
        
        # Solve the system
        sol = np.linalg.solve(LHS_matrix, RHS_vector)
        
        # Extract the velocity components (it's all we care about here)
        u, v, w = sol[[0, n_of_coefficients, n_of_coefficients*2], 0]
        vel = (u, v, w)
        
        # Lastly compute residual
        residual_u = U - np.dot(X, sol[:n_of_coefficients, 0])
        residual_v = V - np.dot(X, sol[n_of_coefficients:2*n_of_coefficients, 0])
        residual_w = W - np.dot(X, sol[2*n_of_coefficients:3*n_of_coefficients, 0])
        residuals = (residual_u, residual_v, residual_w)
        
        return vel, residuals
   
    def _estimate_velocity_at_sphere_center_with_griddata(self, velfield_coor, velfield_values, special=False,
                                                          returnInterp=False, griddata_interp=None, evaluationPoint=np.array([[0, 0, 0]])):
        if griddata_interp is None:
            assert np.ma.size(velfield_values, axis=1) == 3, 'All three components of the velocity field need to be provided.'
            # Initialise a griddata interpolator
            griddata_interp = GridDataInterpolator(velfield_coor)
            
            # Compute the Delaunay triangulation
            griddata_interp.Triangulate()
        
        if returnInterp:
            return griddata_interp
        
        # Compute Barycentric coordinates
        griddata_interp.ComputeBarycentric(evaluationPoint, batchmode=True, special=special, overwrite=True)
        
        # Evaluate the function three times for u, v and w
        u = griddata_interp(velfield_values[:,0])[0]
        v = griddata_interp(velfield_values[:,1])[0]
        w = griddata_interp(velfield_values[:,2])[0]
        
        # Combine results into the velocity vector
        vel_vector = (u, v, w)
        
        return vel_vector
    
    def _sunflower_distribution(self, center, radius, Npoints):
        idcs = np.arange(0, Npoints, dtype=float) + 0.5
        
        r_distribution = np.sqrt(idcs / Npoints) * radius
        theta_distribution = np.pi * (1 + np.sqrt(5)) * idcs
        
        x_distribution = r_distribution * np.cos(theta_distribution) + center[0]
        y_distribution = r_distribution * np.sin(theta_distribution) + center[1]
        z_distribution = np.ones_like(y_distribution) * center[2]
        
        points = np.c_[x_distribution, y_distribution, z_distribution]
        
        return points
    
    def _poissonSampleDistributionCircle(self, center, radius, Npoints, resolution=100):
        # 1. Define the points on the outer circle
        theta = np.linspace(0, 2*np.pi, resolution+1)
        outerPoints = np.c_[center[0] + radius*np.cos(theta[:-1]),
                            center[1] + radius*np.sin(theta[:-1]),
                            center[2] * np.ones(resolution)]
        allPoints = np.r_[outerPoints, np.c_[center].T]
        
        # Define the triangles
        trianglesPoints = [[i, i+1, resolution] for i in range(resolution-1)] + [[resolution-1, 0, resolution]]
        
        # Now we have all the components to create the circular mesh
        npPoints = np.array(allPoints)
        npTriangles = np.array(trianglesPoints).astype(np.int32)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(npPoints)
        mesh.triangles = o3d.utility.Vector3iVector(npTriangles)
        
        # 2. Generate the Poisson sample points
        # Set the seed to control how poisson disk sampling operates
        o3d.utility.random.seed(0)
        # Generate points
        pcd = mesh.sample_points_poisson_disk(Npoints)
        distributedPoints = np.asarray(pcd.points)
        
        return distributedPoints
        
    def _RMS(self, x):
        return np.sum(np.abs(x)) / np.sqrt(len(x))
    
    def _RSS(self, x):
        return np.sum(x**2)
   
    def _getLocalVelfield(self, locator, R, point, velfield):
        localIds = self._find_ids_in_sphere_with_Locator(locator, R, point)
        if len(localIds) > 0:
            return velfield[localIds, :]
        else:
            return None
   
    def _buildTransformedLocalFluidMesh(self, coordinates, uvw, remainder):
        fluidmeshLocalTransformed_vtk = numpy_points_to_vtk_polydata(coordinates)
        velfieldLocalTransformed = np.c_[coordinates, uvw, remainder]
        return fluidmeshLocalTransformed_vtk, velfieldLocalTransformed
   
    def _RodriguesRotation(self, v, k, theta):
        return ( v * np.cos(theta) +
                 np.cross(k, v) * np.sin(theta) +
                 k * np.c_[np.dot(k, v.T)] * (1 - np.cos(theta))
                 )
    
    def _updateGroundEstimate(self, cell, point, funcRotateForward,
                              funcRotateBackward, Rvary, dz_ground,
                              Nvary=6, Rlocal=None, rotateShabang=False,
                              dz0DueToTracerSize=0.25,
                              columnHeight=20, fm=None):    
        
        if (not isinstance(Rlocal, type(None))) and rotateShabang:
            rotateVelfield = True
            shiftVelfieldAfter = False
            # Use a local velfield and translate
            tracerFieldLocal = self._getLocalVelfield(self.pointLocator_groundPlaneTracers,
                                                      Rlocal, point,
                                                      self.groundPlaneTracers_np)
            if isinstance(tracerFieldLocal, type(None)):
                # print('\tNo velfield tracers found')
                return None
            
            # Rotate the cropped tracers field
            NtracersLocal = np.ma.size(tracerFieldLocal, axis=0)
            coorLocalRotated = funcRotateForward(tracerFieldLocal - point)
            uvwLocalRotated = np.zeros((NtracersLocal, 3))
            
            # Build the rotated tracers field
            (fluidmeshLocalRotated_vtk,
             velfieldLocalRotated) = self._buildTransformedLocalFluidMesh(coorLocalRotated,
                                                                          uvwLocalRotated,
                                                                          tracerFieldLocal[:,6:])

            # velfieldToUse = velfieldLocalRotated
            # fluidmeshToUse = fluidmeshLocalRotated_vtk
        elif (not isinstance(Rlocal, type(None))):
            shiftVelfieldAfter = False
            rotateVelfield = False
            # Use a local velocity field
            tracerFieldLocal = self._getLocalVelfield(self.pointLocator_groundPlaneTracers,
                                                      Rlocal, point,
                                                      self.groundPlaneTracers_np)
            self.tracerFieldLocal = tracerFieldLocal 
            print('heeere')
            # Build the local tracers field
            (fluidmeshLocal_vtk,
             velfieldLocal) = self._buildTransformedLocalFluidMesh(tracerFieldLocal - point,
                                                                   np.zeros_like(tracerFieldLocal),
                                                                   tracerFieldLocal[:,6:])
            # velfieldToUse = velfieldLocal
            # fluidmeshToUse = fluidmeshLocal_vtk
        else:
            shiftVelfieldAfter = True
            rotateVelfield = False
            # Use the whole velocity field
            # velfieldToUse = self.groundPlaneTracers_np
            # fluidmeshToUse = fm
        
        # Rotate the point as well
        if shiftVelfieldAfter:
            pointRotated = point    
        else:
            pointRotated = (0., 0., 0.)
        ########################
        ### Create multiple cylinders to check the lowest tracer particle found
        dz0_lowest_tracer_lst = []
        # columns_bot = np.arange(-columnHeight/2, columnHeight/2, Rvary)
        for cyl_i in range(Nvary):
            # Determine position of search radius
            thetai = 2*np.pi / Nvary * cyl_i
            pos = point + np.array([Rvary*np.cos(thetai), Rvary*np.sin(thetai), -columnHeight/2])
            
            # fluidmeshCylinderVaryIds = self._find_ids_in_column_with_Cylinder(fluidmeshToUse, Rvary,
            #                                                                     pos, columnHeight,
            #                                                                     np.array([0., 0., 1.])
            #                                                                     )
            # for col_bot in columns_bot:
            #     enclosedPointsCylinderVary = vtk.vtkIdList()
            #     self.pointLocator_groundPlaneTracers.FindPointsWithinRadius(Rvary,
            #                                                                 pointRotated,
            #                                                                 enclosedPointsCylinderVary)
            #     fluidmeshCylinderVaryIds = self._vtk_idlist_to_lst(enclosedPointsCylinderVary)
            
            
            enclosedPointsCylinderVary = vtk.vtkIdList()
            self.pointLocator_groundPlaneTracers.FindPointsWithinRadius(Rvary,
                                                                        pos,
                                                                        enclosedPointsCylinderVary)
            fluidmeshCylinderVaryIds = self._vtk_idlist_to_lst(enclosedPointsCylinderVary)
            self.fluidmeshCylinderVaryIds = fluidmeshCylinderVaryIds 
            
            if len(fluidmeshCylinderVaryIds) > 0:
                # Extract the velocity field values using the mask
                # velfieldCylinderVary = self.velfieldToUse[fluidmeshCylinderVaryIds, :]
                velfieldCylinderVary = self.groundPlaneTracers_np[fluidmeshCylinderVaryIds, :]
                if shiftVelfieldAfter:
                    velfieldCylinderVary[:,:3] -= point
                    # velfieldCylinderVary = np.array([velfieldToUse[i, :] - point for i in fluidmeshCylinderVaryIds])
                # else:
                #     velfieldCylinderVary = np.array([velfieldToUse[i, :] for i in fluidmeshCylinderVaryIds])
                
                # Sort the points
                velfieldCylinderVarySorted = velfieldCylinderVary[velfieldCylinderVary[:,2].argsort()]
                
                # Add lowest point estimate
                lowestTracerVary = velfieldCylinderVarySorted[0,:]
                dz0_lowest_tracer = lowestTracerVary[2] - pointRotated[2]
                dz0_lowest_tracer_lst.append(dz0_lowest_tracer)
                
                
                # if rotateVelfield:
                #     newGroundPoint = funcRotateBackward(newGroundPointRotated)
                #     shift = newGroundPoint - pointRotated
                # else:
                #     shift = newGroundPointRotated - pointRotated
            else:
                dz0_lowest_tracer_lst.append(np.nan)
        
        if np.any(dz0_lowest_tracer_lst == np.nan):
            shift = None
        else:
            dz0_mean = np.mean(dz0_lowest_tracer_lst)
            newGroundPointRotated = np.array([pointRotated[0], pointRotated[1], pointRotated[2] + dz0_mean - dz0DueToTracerSize])
            
                    # 3. Rotate the new Ground point backward to original coordinate system
            if rotateVelfield:
                newGroundPoint = funcRotateBackward(newGroundPointRotated)
                shift = newGroundPoint - pointRotated
            else:
                shift = newGroundPointRotated - pointRotated
        
        return shift
   
    def AddVelocityField_Pandas(self, velfieldPD):
        self.AddVelocityField_Numpy(velfieldPD.to_numpy())
        self.velfieldPD = velfieldPD
        return
   
    def AddVelocityField_Numpy(self, velfield, columns: list = []):
        assert len(np.shape(velfield)) == 2, 'Provide 2D array (NxM) of velfield information, with x, y, z, u, v, w, rest...'
        assert np.ma.size(velfield, axis=1) >= 6, 'Provide at least 6 columns for the velocity field in the following order: X, Y, Z, U, V, W, rest...'
        
        self.velfield = velfield
        self.NExtraVelfieldValues = np.ma.size(self.velfield, axis=1) - 3
        if len(columns) > 0:
            self.velfieldPD = pd.DataFrame(velfield, index=None, columns=columns)
            
            # Check which name to use for vorticity
            for vorticityNameX in self.vorticityNameXOptions:
                if vorticityNameX.upper() in [column.upper() for column in self.velfieldPD.columns]:
                    self.vorticityNameX = vorticityNameX
                    break
                
            # Check which name to use for vorticity Y
            for vorticityNameY in self.vorticityNameYOptions:
                if vorticityNameY.upper() in [column.upper() for column in self.velfieldPD.columns]:
                    self.vorticityNameY = vorticityNameY
                    break
                
            # Check which name to use for vorticity Z
            for vorticityNameZ in self.vorticityNameZOptions:
                if vorticityNameZ.upper() in [column.upper() for column in self.velfieldPD.columns]:
                    self.vorticityNameZ = vorticityNameZ
                    break
                
            if hasattr(self, 'vorticityNameX'):
                self.omegaColumns = [self.vorticityNameX,
                                     self.vorticityNameY,
                                     self.vorticityNameZ]
        pass
        
    def BuildFluidMeshFromVelocityField(self):
        self.AddFluidMesh_Numpy(self.velfield[:,:3])
        pass
    
    def AddGroundPlaneTracers(self, tracers):
        # Save the tracers
        self.groundPlaneTracers_np = tracers
        self.groundPlaneTracers_vtk = numpy_points_to_vtk_polydata(tracers)
        self.Ntracers = self.groundPlaneTracers_vtk.GetNumberOfPoints()
        pass
    
    def AddFluidMesh_Numpy(self, fluidmesh_np):
        self.fluidmesh_np = fluidmesh_np
        self.fluidmesh_vtk = numpy_points_to_vtk_polydata(fluidmesh_np)
        
        # add ids to fluidmesh_vtk to simplify tracking later
        ids_array = vtk.vtkIntArray()
        ids_array.SetNumberOfComponents(1)
        ids_array.SetNumberOfTuples(self.fluidmesh_vtk.GetNumberOfPoints())
        ids_array.SetName('ids')
        for i in range(self.fluidmesh_vtk.GetNumberOfPoints()):
            ids_array.InsertTuple(i, [i])
            
        self.fluidmesh_vtk.GetPointData().AddArray(ids_array)
        pass
        
    def AddObjectMesh_VTK(self, objectmesh_vtk):
        self.objectmesh_vtk = objectmesh_vtk
        self.objectmesh_np = np_nodal_coordinates_from_unstructured_vtk_grid(objectmesh_vtk)
        pass
    
    def AddModelPolyData_VTK(self, modelPD, maxArea=25., maxEdgeLength=25.):
        self.modelPD_vtk = modelPD
        # Add an upsampled version
        print('Adding upsampled Version of model PolyData')
        ASF = vtk.vtkAdaptiveSubdivisionFilter()
        ASF.SetInputData(modelPD)
        ASF.SetMaximumTriangleArea(maxArea)
        ASF.SetMaximumEdgeLength(maxEdgeLength)
        ASF.Update()
        self.modelPD_vtkUpSampled = ASF.GetOutput()
        print('Finished upsampling')
        print('.')
        print('.')
        pass
    
    def AddOutputMesh_VTK(self, outputmesh_vtk):
        self.outputmesh_vtk = outputmesh_vtk
        self.outputmesh_np = np_nodal_coordinates_from_unstructured_vtk_grid(outputmesh_vtk)
        self.Npoints = outputmesh_vtk.GetNumberOfPoints()
        self.Ncells = outputmesh_vtk.GetNumberOfCells()
        
        # Allocate an array for the dz-shift
        if self.updateGroundEstimate:
            self.shiftVector = np.zeros((self.Ncells, 3))
        else:
            self.shiftVector = None
        pass
    
    def BuildPointLocators(self, points_per_bucket=100):
        assert hasattr(self, 'fluidmesh_vtk'), 'Can not build vtkPointLocators, call AddFluidMesh_Numpy() first.'
        # assert hasattr(self, 'objectmesh_vtk'),  'Can not build vtkPointLocators, call AddObjectMesh_VTK() first.'

        # create pointLocator object for output mesh points
        self.pointLocator_outputmesh = self._build_vtkPointLocator(self.outputmesh_vtk, points_per_bucket)

        # create pointLocator object for fluid mesh points
        self.pointLocator_fluidmesh = self._build_vtkPointLocator(self.fluidmesh_vtk, points_per_bucket)

        # Create a pointLocator object for tracer particles
        if self.updateGroundEstimate and hasattr(self, 'groundPlaneTracers_vtk'):
            self.pointLocator_groundPlaneTracers = self._build_vtkPointLocator(self.groundPlaneTracers_vtk, points_per_bucket)

        # Create a cell locator for the polydata                
        if hasattr(self, 'modelPD_vtk'):
            self.cellLocator_objectSTL = self._build_vtkCellLocator(self.modelPD_vtk)
            
    def _VolumeCutSphere(self, R, h):
        return np.pi/3 * (4*R**3 + h**3   - 3*R*h**2)
    
    def _estimateLinearMeanInCoin(self, center, velfieldCoor, velfieldValInCoin,
                            Np, projectedPointTranslated=None, returnGradient=False):
        # Estimate linearised mean using LSR with variation in x, y and z
        X = np.c_[np.ones(Np),
                  velfieldCoor[:,0],
                  velfieldCoor[:,1],
                  velfieldCoor[:,2]]
        Y = np.c_[velfieldValInCoin]
        if projectedPointTranslated is not None:
            # print(f'Translated projected point to ({np.round(projectedPointTranslated, 3)})\n')
            
            # Define constraint
            C = np.c_[np.ones(1),
                      projectedPointTranslated[0],
                      projectedPointTranslated[1],
                      projectedPointTranslated[2], ]
            # C = np.c_[np.ones(1), projectedPointTranslated[2], ]
            d = np.c_[0]
            
            # Solve LSR problem
            solDict = pivOpt.LSR_with_constraints(X, Y, C, d)
            
            # Extract output
            sol = solDict['x']
        else:
            sol = self._least_squares_solve(X, Y)
        
        # Extract mean velocity
        meanVelocity = sol[0,0]
        gradient = sol[3,0]
        
        # Compute residual
        residual = Y - np.dot(X, sol)
        
        if returnGradient:
            return meanVelocity, residual, gradient
        else:
            return meanVelocity, residual
    
    def _estimateLinearMeanInCylinder(self, point, velfieldCoor, velfieldValInCoin,
                            Np, projectedPointTranslated=None):
        # Estimate linearised mean using LSR with variation in z only
        X = np.c_[np.ones(Np),
                  velfieldCoor[:,2]]
        Y = np.c_[velfieldValInCoin]
        if projectedPointTranslated is not None:
            # print(f'Translated projected point to ({np.round(projectedPointTranslated, 3)})\n')
            
            # Define constraint
            C = np.c_[np.ones(1),
                      projectedPointTranslated[2], ]
            # C = np.c_[np.ones(1), projectedPointTranslated[2], ]
            d = np.c_[0]
            
            # Solve LSR problem
            solDict = pivOpt.LSR_with_constraints(X, Y, C, d)
            
            # Extract output
            sol = solDict['x']
        else:
            sol = self._least_squares_solve(X, Y)
        
        # Extract mean velocity
        # centerVelocity = sol[0,0]
        gradient = sol[1,0]
        velAtPoint = np.dot(np.r_[1, point[2]], sol)[0]
        
        # Compute residual
        residual = Y - np.dot(X, sol)
        
        return velAtPoint, gradient, residual
    
    def ComputeGroundPlaneFit(self, dz_ground=0, Rvary=12,
                              minDistanceToObject=12, Rlocal=None,
                              track_percentage=False, partitions=100,
                              recompute=False, savefolder='',
                              trackProgressStream=None):
        # Check if progress needs to be streamed to another thread
        streamProgress = not isinstance(trackProgressStream, type(None))
        
        # 1. Check if the ground plane fit was already computed and could be loaded instead of computed
        currentFileName = f'Ntracers={self.Ntracers}_Ncells={self.Ncells}_dz={dz_ground}_Rvary={Rvary}_minDist={minDistanceToObject}_Rlocal={Rlocal}.npy'
        if savefolder != '' and (not recompute):
            print('')
            print('Trying to find previously computed file for updating ObjectRegistration estimate')
            #1. Extract all files in the folder
            filesInDir = os.listdir(savefolder)
            
            #2. Check if current settings coincide with one of the files
            oldFilePresent = currentFileName in filesInDir
            if oldFilePresent:
                print('Previous file found, loading it in...')
            else:
                print('No previous file found, will be saved after computation...')
        else:
            oldFilePresent = False
        
        if oldFilePresent:
            #1. Load file
            dataArray = np.load((Path(savefolder) / currentFileName).as_posix(), allow_pickle=True)
            
            # Load the needed attributes
            # order: Original points, isShiftComputed, local shift, fitted shift
            self.originalPoints = dataArray[:, :3]
            self.shiftComputed = dataArray[:, 3].astype(bool)
            self.shiftVector = dataArray[:, 4:7]
            self.shiftedPointsFit = dataArray[self.shiftComputed, 6]
            self.shiftedPoints = dataArray[:, 7:10] + self.originalPoints
            
            print('GroundPlaneFit updated')
            
        else:
            # Initialise an array to track if all shifts have been computed
            self.shiftComputed = np.zeros((self.Ncells,), dtype=bool)
            self.originalPoints = np.zeros((self.Ncells, 3))
            
            percent_count = 1
            Ncells_frac = self.Ncells/partitions
            # Loop over all cells to estimate the ground shift
            numberOfShiftedPoints = 0
            for c_idx in range(self.Ncells):
            # for c_idx in range():
            
                if track_percentage and (c_idx // (percent_count * Ncells_frac) == 1):
                    if streamProgress:
                        trackProgressStream(round(percent_count*100/partitions))
                    else:
                        print(f'{(percent_count*100/partitions):.1f}% DONE...')
                    percent_count += 1
                
                # 1. Get the cell and center of the cell
                vtk_cell = self.outputmesh_vtk.GetCell(c_idx)
                vtk_point = getVTKCellCenter(vtk_cell)
                self.originalPoints[c_idx, :] = vtk_point
                
                # Compute cell normal
                cellNormal = np.array([0., 0., 0.])
                cellPoints = vtk_cell.GetPoints()
                vtk_cell.ComputeNormal(cellPoints.GetPoint(0), cellPoints.GetPoint(1), cellPoints.GetPoint(2), cellNormal)
                
                # Compute distance to object
                closest_point = np.array([0.0, 0.0, 0.0])
                cell_obj = vtk.vtkGenericCell()
                projected_cellId_obj = vtk.reference(0)
                subId_obj = vtk.reference(0)
                dist2_obj = vtk.reference(0.)
                
                self.cellLocator_objectSTL.FindClosestPoint(vtk_point,
                                                            closest_point,
                                                            cell_obj,
                                                            projected_cellId_obj,
                                                            subId_obj,
                                                            dist2_obj
                                                            )
                distanceToObject = np.sqrt(dist2_obj.get())
                
                if distanceToObject <= minDistanceToObject:
                    # print('skip')
                    continue
                
                # Determine functions to rotate the whole shabang to and from
                # cellNormal using Rodrigues' rotation
                # (This is needed to more easily place the cylinders)
                upVector = np.array([0., 0., 1.])
                if np.linalg.norm(cellNormal - upVector) < 1e-5:
                    # If the normal direction is already pointing up, we don't need to rotate anything
                    funcRotateForward = lambda vec: vec
                    funcRotateBackward = lambda vec: vec
                    rotationAngle = 0.
                else:
                    rotationAxis = np.cross(cellNormal, upVector)
                    rotationAxisNormalized = rotationAxis / np.linalg.norm(rotationAxis)
                    rotationAngle = np.arccos(np.dot(cellNormal, upVector))
                
                    # Use Rodrigues' rotation formula
                    funcRotateForward = lambda vec: self._RodriguesRotation(vec, rotationAxisNormalized, rotationAngle)
                    funcRotateBackward = lambda vec: self._RodriguesRotation(vec, rotationAxisNormalized, -1*rotationAngle)
                
                shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                              Rvary, dz_ground, Rlocal=Rlocal, fm=self.fluidmesh_vtk)
                
                if isinstance(shiftVectorLocal, type(None)):
                    continue
                # Save the results
                if not np.any(np.isnan(shiftVectorLocal)):
                    self.shiftVector[c_idx, :] = shiftVectorLocal
                    self.shiftComputed[c_idx] = True
                    numberOfShiftedPoints += 1
                
            # When all shifts have been computed, fit a plane through points
            shiftedPoints = self.originalPoints[self.shiftComputed, :]
            shiftedPoints[:,2] += self.shiftVector[self.shiftComputed, 2]
            
            
            
            # Change the ground points into a scikit-spatial Points instance
            shiftedPointsSKSpatial = Points(shiftedPoints)
            # Compute the best fit for a ground plane
            groundPlaneFit = Plane.best_fit(shiftedPointsSKSpatial )
            # Extract the plane components a, b, c and compute d
            self.planeFitNormal = groundPlaneFit.normal
            a, b, c = groundPlaneFit.normal
            xPLaneCentroid, yPLaneCentroid, zPLaneCentroid = groundPlaneFit.point
            d = -a*xPLaneCentroid - b*yPLaneCentroid - c*zPLaneCentroid
            
            # Now for each shiftedPoint, compute the fitted z-location based on the x and y
            shiftedPointsFitZFunc = lambda x, y: (-d - a*x - b*y)/c
            shiftedPointsFitZ = shiftedPointsFitZFunc(shiftedPointsSKSpatial[:,0], shiftedPointsSKSpatial[:,1])
            
            # Lastly save the fitted points
            self.shiftedPointsFit = np.c_[shiftedPointsSKSpatial[:,0:2], shiftedPointsFitZ]
            self.shiftedPoints = self.originalPoints.copy()
            self.shiftedPoints[self.shiftComputed, :] = self.shiftedPointsFit
        
        self.groundPlaneFitComputed = True
        
        if (recompute and (savefolder != '')) or (savefolder != '' and (not recompute) and (not oldFilePresent)):
            # Save the data
            print(f'Saving the groundPlaneFit for later use as:\n{currentFileName}')
            
            # 1. Construct the data array
            # order: Original points, isShiftComputed, local shift, fitted shift
            dataArray = np.c_[self.originalPoints, self.shiftComputed, self.shiftVector, self.shiftedPoints-self.originalPoints]
            
            # 2. save the data array
            np.save((Path(savefolder) / currentFileName).as_posix(), dataArray)
            
        return
    
    def AddConcentration(self, C):
        self.C = C
        pass
    
    def _NOMPoints_from_NFMPoints(self, Np, rCircleEq):
        area = rCircleEq**2 * np.pi
        return round(area * self.C)
        # volume = Np / self.C
        # self.volumeSphere = volume
        # rVolumeEq = (3/4 * volume / np.pi)**(1/3)
        # self.rSphereEq = rVolumeEq
        
        # return round( Np * (rCircleEq / rVolumeEq)**2/3)
    
    def AddSlicer(self, slicer_option, r=0):
        if slicer_option == 'plane':
            self.sliceFunc = lambda x, y, z: z >= 0.
        elif slicer_option == 'sphere':
            self.sliceFunc = lambda x, y, z: z >= r - np.sqrt(r**2 - x**2 - y**2)
        else:
            raise ValueError(f'Incorrect choice of slicer option "{slicer_option}"')
            
        self.sliceFuncAdded = True
        
    def Run(self, track_percentage=False, div_free=False, test=False, partitions=10,
            special=False, use_all_constraints=False, dz_ground=0,
            Rvary=12, Rlocal=50, minDistanceToObject=12,
            stepSizeAlongNormal = 1e-1, maxNStepsAlongNormal=400,
            omegaThreshold=100, c_start=0, findBL=False, trackProgressStream = None):
        # Note the flag special slightly changes how the griddata barycentric coordinates are computed. When this is set to True
        # the program will try to determine if a point still lies outside the convex hull, if it is moved a slight distance in
        # either the x, y, or z-direction. If the point lies inside the convex hull after moving, this is chosen as the new location
        # for the point to compute the barycentric coordinates.
        
        assert hasattr(self, 'pointLocator_fluidmesh'), 'Can not Run iteration, call BuildPointLocators() first.'
        
        streamProgress = not isinstance(trackProgressStream, type(None))
        
        self.div_free = div_free
        
        # Initialize xyz, u, v and w
        self.xyz = np.zeros((self.Ncells, 3))
        self.u = np.zeros((self.Ncells,))
        self.v = np.zeros((self.Ncells,))
        self.w = np.zeros((self.Ncells,))
        
        if hasattr(self, 'vorticityNameX') and hasattr(self, 'vorticityNameY') and hasattr(self, 'vorticityNameZ'):
            self.omegaX = np.zeros((self.Ncells,))
            self.omegaY = np.zeros((self.Ncells,))
            self.omegaZ = np.zeros((self.Ncells,))
            
        # Initialize the gradients
        self.dudx = np.zeros((self.Ncells,))
        self.dvdx = np.zeros((self.Ncells,))
        self.dwdx = np.zeros((self.Ncells,))
        
        self.dudy = np.zeros((self.Ncells,))
        self.dvdy = np.zeros((self.Ncells,))
        self.dwdy = np.zeros((self.Ncells,))
        
        self.dudz = np.zeros((self.Ncells,))
        self.dvdz = np.zeros((self.Ncells,))
        self.dwdz = np.zeros((self.Ncells,))
        
        self.omega = np.zeros((self.Ncells, 3))
        
        self.BLvalid = np.zeros((self.Ncells,), dtype=bool)
        self.delta = np.zeros((self.Ncells,))
        self.deltaStar = np.zeros((self.Ncells,))
        self.theta = np.zeros((self.Ncells,))
        self.H = np.zeros((self.Ncells,))
        self.velWallNormalGradientInPlane = np.zeros((self.Ncells, 3))
        
        self.isValid = np.zeros((self.Ncells,), dtype=bool)
                
        # Initialize some properties to keep a histogram with
        # 0: Number of fluidmesh points in sphere
        # 1: Number of objectmesh points in sphere
        # 2: RMS of LSR residual for u
        # 3: RMS of LSR residual for v
        # 4: RMS of LSR residual for w
        # 5: RMS of LSR residual for u at object mesh nodes
        # 6: RMS of LSR residual for v at object mesh nodes
        # 7: RMS of LSR residual for w at object mesh nodes
        # 8: RSS of LSR regression for u
        # 9: RSS of LSR regression for v
        # 10:RSS of LSR regression for w
        # 11:Reynolds Stress u
        self.info = np.full((self.Ncells, 12), np.nan)
        self.info[:,:2] = 0
        
        # Initialise average number of fluidmesh points        
        self.avg_NFM = 0
        
        # Add some variables to track percentage done
        percent_count = 1
        Ncells_frac = int(self.Ncells/partitions)
        
        N_timesteps = 1000
        time_lst = np.zeros((N_timesteps,))
# =============================================================================
#         # Walk over all cells to perform the least squares regression
# =============================================================================
        for c_idx in range(c_start, self.Ncells):
        # for c_idx in range(c_start, 500):
            self.c_idx = c_idx
            if test:
                t0 = time.time()
            # 0. Log percentage done if option given
            if track_percentage and (c_idx // ((percent_count) * Ncells_frac) == 1):
                if streamProgress:
                    trackProgressStream(round(percent_count*100/partitions))
                else:
                    print(f'{round(percent_count*100/partitions)}% DONE...')
                percent_count += 1
                
# =============================================================================
#                 # AA. Set up the loop by extracting general data such as:
#                 #   - cell/points coordinates and normals
#                 #   - Rotation functions
# =============================================================================
            # 1. Get the cell and center of the cell
            vtk_cell, vtk_point, cellNormal = self._get_cell_info(c_idx)
            
            self.xyz[c_idx, :] = vtk_point
            
            if test:
                print(f'Processing cell {c_idx} with center at {np.round(vtk_point,3)}')
            
            # Compute distance to object
            _, dist2_obj = self.FindClosestPointToObject(vtk_point)
            distanceToObject = np.sqrt(dist2_obj)
            
            # Determine functions to rotate the whole shabang to and from
            # cellNormal using Rodrigues' rotation
            upVector = np.array([0., 0., 1.])
            (funcRotateForward,
             funcRotateBackward,
             rotationAngle,
             rotMat) = self.GetRotationInfo(cellNormal, upVector)
            
            
                        
            if test:
                print(f'\tRotationAngle is {rotationAngle*180/np.pi:.2f} degrees')
            
# =============================================================================
#             # BB. Rotate the velocity to have the outward
#                       normal pointing in the positive (local) z-direction
#                       and localise velocity field if needed.
# =============================================================================
            if self.velMethod == '0' or findBL:
                # In this case we find a local velocity field first which we
                # speed up the computations (does it actually speed up...?)
                
                # The local velocity field is shifted and rotated
                
                if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
                    if test:
                        print('\tLocating shift in object location for current cell')
                    self._find_ground_shift_for_cell(c_idx, vtk_point,
                                                     vtk_cell, distanceToObject,
                                                     funcRotateForward, funcRotateBackward,
                                                     Rvary, dz_ground, Rlocal
                                                     )
                    # Determine the shift in ground plane
                    if self.groundPlaneFitComputed:
                        shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
                    else:
                        shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                                      Rvary, dz_ground, Rlocal=Rlocal)
                        
                        # Add result to shiftVector array
                        if shiftVectorLocal is None:
                            continue
                        self.shiftVector[c_idx, :] = shiftVectorLocal
                    
                    if test:
                        print(f'\tLocal shift computed to be {np.round(shiftVectorLocal, 3)} mm in [x y z]')
                else:
                    if test:
                        print('\tNo shift computed for current cell.')
                    
                    shiftVectorLocal = np.zeros(3,)
                # Use a localized velocity field and shift the velocity field
                velfieldLocal = self._getLocalVelfield(self.pointLocator_fluidmesh, Rlocal, vtk_point, self.velfield)
                coorLocalTranslated = velfieldLocal[:,:3] - shiftVectorLocal
                uvwLocal = velfieldLocal[:,3:6]
                
                # Crop the coordinates if they fall inside the ground
                maskCoordinatesLocal = coorLocalTranslated[:,2] >= 0
                
                coorLocalTransformedCropped = funcRotateForward(coorLocalTranslated[maskCoordinatesLocal, :] - vtk_point)
                uvwLocalCropped = uvwLocal[maskCoordinatesLocal, :]
                
                # Build the translated velocity field
                (fluidmeshLocalTranslated_vtk,
                 velfieldLocalTranslated) = self._buildTransformedLocalFluidMesh(coorLocalTransformedCropped,
                                                                                 uvwLocalCropped,
                                                                                 velfieldLocal[maskCoordinatesLocal,6:]
                                                                                 )

                # Update the fluidmesh point locator
                applyRotationAfterwards = False
                fluidmeshToUse = fluidmeshLocalTranslated_vtk
                velfieldToUse = velfieldLocalTranslated
                self.velfieldToUse = velfieldLocalTranslated
                self.shift = shiftVectorLocal
                self.angle = rotationAngle
                
                # Build a pointLocator for the local fluid mesh
                pointLocator_fluidmeshToUse = self._build_vtkPointLocator(fluidmeshToUse, 100)
                
            elif self.useCoin:
                # In this case, we use a local velocity field, but do not build
                # a special locator for each cell, since we don't use that
                
                # The velocity field is subsequently shifted and rotated
                if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
                    if test:
                        print('\tLocating shift in object location for current cell')
                    # Determine the shift in ground plane
                    if self.groundPlaneFitComputed:
                        shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
                    else:
                        shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                                      Rvary, dz_ground, Rlocal=Rlocal)
                        
                        # Add result to shiftVector array
                        if shiftVectorLocal is None:
                            continue
                        self.shiftVector[c_idx, :] = shiftVectorLocal

                else:
                    shiftVectorLocal = np.zeros(3,)

                applyRotationAfterwards = True
                fluidmeshToUse = self.fluidmesh_vtk
                velfieldToUse = self.velfield
                pointLocator_fluidmeshToUse = self.pointLocator_fluidmesh
                    
                # # Use a localized velocity field and shift the velocity field
                # RlocalField = 5*np.sqrt((self.coinHeight * 2)**2  + (self.R)**2)
                # RlocalCenter = vtk_point +  cellNormal * self.coinHeight * 2
                # velfieldLocal = self._getLocalVelfield(self.pointLocator_fluidmesh,
                #                                        RlocalField,
                #                                        RlocalCenter,
                #                                        self.velfield)
                # if isinstance(velfieldLocal, type(None)):
                #     continue
                # coorLocalTranslated = velfieldLocal[:,:3] - shiftVectorLocal
                # uvwLocal = velfieldLocal[:,3:6]
                
                # # Crop the coordinates if they fall inside the ground
                # maskCoordinatesLocal = coorLocalTranslated[:,2] >= 0
                
                # coorLocalTransformedCropped = funcRotateForward(coorLocalTranslated[maskCoordinatesLocal, :] - vtk_point)
                # uvwLocalCropped = uvwLocal[maskCoordinatesLocal, :]
                
                # # Build the translated velocity field
                # (fluidmeshLocalTranslated_vtk,
                #  velfieldLocalTranslated) = self._buildTransformedLocalFluidMesh(coorLocalTransformedCropped,
                #                                                                  uvwLocalCropped,
                #                                                                  velfieldLocal[maskCoordinatesLocal,6:]
                #                                                                  )

                # # Update the fluidmesh point locator
                # applyRotationAfterwards = False
                # velfieldToUse = velfieldLocalTranslated
                # fluidmeshToUse = fluidmeshLocalTranslated_vtk
                # self.velfieldToUse = velfieldLocalTranslated
                # self.shift = shiftVectorLocal
                # self.angle = rotationAngle
                
                
            else:
                # In this case we do not change the velocity field.
                # Only the ground-shift is determined, which is later used to
                # shift the velocity field when fluidmesh nodes inside the
                # local sphere are considered
                if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
                    if test:
                        print('\tLocating shift in object location for current cell')
                    
                    # Determine the shift in ground plane
                    if self.groundPlaneFitComputed:
                        shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
                    else:
                        shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                                      Rvary, dz_ground, Rlocal=Rlocal)
                        
                        # Add result to shiftVector array
                        if shiftVectorLocal is None:
                            continue
                        self.shiftVector[c_idx, :] = shiftVectorLocal
                    
                    if test:
                        print(f'\tLocal shift computed to be {np.round(shiftVectorLocal, 3)} mm in [x y z]')
                else:
                    if test:
                        print('\tNo shift computed for current cell.')

                    shiftVectorLocal = np.zeros(3,)

                applyRotationAfterwards = True
                fluidmeshToUse = self.fluidmesh_vtk
                velfieldToUse = self.velfield
                pointLocator_fluidmeshToUse = self.pointLocator_fluidmesh
            
            # Construct the velocity field to use as pandas dataframe
            if hasattr(self, 'velfieldPD') and self.useSphere:
                velfieldToUsePD = pd.DataFrame(velfieldToUse,
                                               index=None,
                                               columns=self.velfieldPD.columns
                                               )

# =============================================================================
#             # CC. Determine the velocity field nodes which lie inside the
#             #       volume of interest.
# =============================================================================

            # 2. Find all nodes inside a sphere with radius R around the point
            if self.useSphere:
                # In the case of the extrapolation, we use a cylinder of points
                # since it is not possible to determine the number of steps
                # outward to take before interpolation is succesful.
                if self.velMethod == '0':
                    maxH = stepSizeAlongNormal  * maxNStepsAlongNormal * 1.5
                    np_fluidmeshpoints_ids = self._find_ids_in_column_with_Cylinder(fluidmeshToUse,
                                                                                    self.R,
                                                                                    np.array([0., 0., 0.]),
                                                                                    maxH,
                                                                                    upVector)
                else:
                    # When the boundary layer edge must be determined we use
                    # a larger column of points to operate on
                    if findBL:
                          np_fluidmeshpoints_ids = self._find_ids_in_sphere_with_Locator(pointLocator_fluidmeshToUse,
                                                                                         self.R,
                                                                                         np.array([0., 0., 0.])
                                                                                         )
                    else:          
                        np_fluidmeshpoints_ids = self._find_ids_in_sphere_with_Locator(pointLocator_fluidmeshToUse,
                                                                                       self.R,
                                                                                       vtk_point + shiftVectorLocal
                                                                                       )
            else:
                NCoins = 2
                cylinderHeight = self.coinHeight + self.coinHeight * (NCoins-1) * (1 - self.coinOverlap/100)
                dzCoin = (cylinderHeight - self.coinHeight) / (NCoins - 1)
                cylinderbottom = vtk_point + shiftVectorLocal - self.coinHeight/2 * cellNormal
                np_fluidmeshpoints_ids = self._find_ids_in_column_with_Cylinder(fluidmeshToUse,
                                                                                self.R,
                                                                                cylinderbottom,
                                                                                cylinderHeight,
                                                                                cellNormal,
                                                                                rotMat)

            if (self.velMethod == '1') and findBL:
                maxH = stepSizeAlongNormal  * maxNStepsAlongNormal * 1.5
                np_fluidmeshpointsColumn_ids = self._find_ids_in_column_with_Cylinder(fluidmeshToUse,
                                                                                      self.R,
                                                                                      np.array([0., 0., 0.]),
                                                                                      maxH,
                                                                                      upVector)

            # Apply slicer
            # Update velfield in sphere
            velcoor_to_slice = velfieldToUse[np_fluidmeshpoints_ids, :]
            
            if applyRotationAfterwards:
                # 1. Apply shift
                velcoor_to_slice[:,:3] -= shiftVectorLocal
                # 2. Mask all points below 0.
                zmask = velcoor_to_slice[:,2] >= 0.
                velcoor_to_slice = velcoor_to_slice[zmask , :]
                np_fluidmeshpoints_ids = np.array(np_fluidmeshpoints_ids)[zmask]
                # 3. Rotate the true velfield as needed
                velcoor_to_slice[:,:3] = funcRotateForward(velcoor_to_slice[:,:3] - vtk_point)
            
            velcoorMask = self.sliceFunc(*velcoor_to_slice[:,:3].T)
            np_fluidmeshpoints_ids = np_fluidmeshpoints_ids[velcoorMask]
            velfield_in_sphere = velcoor_to_slice[velcoorMask, :]

            NFMPoints = len(np_fluidmeshpoints_ids)
            self.avg_NFM += NFMPoints / self.Ncells
            
            
        
            if self.useCoin and not self.coinFittingMethod=='LIN':
                # Divide this up in to the given number of coins
                FluidMeshPointIdsInCoin = []
                for coini in range(NCoins):
                    velfieldValuesToUse = velfieldToUse[np_fluidmeshpoints_ids, 2]
                    lowBound = coini * dzCoin - self.coinHeight/2
                    highBound = coini * dzCoin + self.coinHeight/2
                    print(f'{lowBound = } | {highBound =}')
                    FluidMeshPointIdsInCoin.append(np_fluidmeshpoints_ids[(velfieldValuesToUse >= lowBound) &
                                                                          (velfieldValuesToUse < highBound)
                                                                          ]
                                                    )
                    
            if self.useSphere:
                if self.griddata_mode:
                    if NFMPoints == 0:
                        continue
                elif NFMPoints < 10:
                    # print(f'\Insufficient fluid mesh nodes found in sphere around cell {c_idx}')
                    continue
            elif self.useCoin:
                if (self.coinFittingMethod=='LIN') and NFMPoints < 100:
                    continue
                if (self.coinFittingMethod!='LIN') and len(FluidMeshPointIdsInCoin[0]) < 100:
                    continue
            
            
# =============================================================================
#             # DD. Determine the object mesh nodes which might require to be added
#             #       to the array of data points
# =============================================================================
            
            if self.useCoin:
                if self.useObjectInfo:
                    objmeshpoints = np.array([0., 0., 0.])
                    NOMPoints = 1
                else:
                    NOMPoints = 0
            else:
                # If needed, find number of object mesh points and its distribution
                if self.useObjectInfo and (not self.LSR_constrained) and self.useSphere:
                    # Find the section of the circle which
                    # 1. Intersects with the ground
                    ground_intersect = self._find_ground_intersect(vtk_point,
                                                                   self.R,
                                                                   z_ground=0.)
                    
                    # 2. Intersects with the model
                    # Check if sphere intersects with object itself
                    closest_point = np.array([0.0, 0.0, 0.0])
                    projected_cellId_obj = vtk.reference(0)
                    subId_obj = vtk.reference(0)
                    dist2_obj = vtk.reference(0.)
                    cell_obj = vtk.vtkGenericCell()
                    model_intersect = vtk.reference(0)
                    
                    self.cellLocator_objectSTL.FindClosestPointWithinRadius(vtk_point,
                                                                            self.R,
                                                                            closest_point,
                                                                            cell_obj,
                                                                            projected_cellId_obj,
                                                                            subId_obj,
                                                                            dist2_obj,
                                                                            model_intersect
                                                                            )
                    
                    # Check if closer to ground or model
                    closerToGround = (not model_intersect) or (vtk_point[2] <= np.sqrt(dist2_obj.get()))
                    
                    if (ground_intersect is not None) and closerToGround:
                        Vs = self._VolumeCutSphere(self.R, self.R - vtk_point[2])                    
                        # Relation 4 for determining Num OM-points
                        # For griddata_mode, we simply use 3 points
                        if self.griddata_mode:
                            NOMPoints = 3
                        else:
                            NOMPoints = self._NOMPoints_from_NFMPoints(NFMPoints, ground_intersect[1])
                            # round(NFMPoints * ground_intersect[1] / (Vs *3 / (4*np.pi))**(1/3))
                        
                        # Using a poisson distribution, place N points inside the circle of intersection
                        groundCenter = np.array([vtk_point[0], vtk_point[1], vtk_point[2]])
                        groundCircleRadius = self.R
                        
                        # Make a switch.
                        # a. When only the ground plane is cut, then use a sunflower distribution
                        if not model_intersect:
                            objmeshpoints = self._sunflower_distribution(groundCenter,
                                                                         groundCircleRadius,
                                                                         NOMPoints)
                        # b. Else use a poisson Sampling distribution
                        else:
                            objmeshpoints = self._poissonSampleDistributionCircle(groundCenter,
                                                                                  groundCircleRadius,
                                                                                  NOMPoints)
                        
                        self.objmeshpoints = objmeshpoints
                    elif model_intersect:
                        sphereCenter = np.array(vtk_point)
                        
                        # When in griddata mode, only three points of the the cell are
                        # needed to define the objectmeshpoints
                        if self.griddata_mode:
                            objmeshpoints = numpy_support.vtk_to_numpy(vtk_cell.GetPoints().GetData())
                            NOMPoints = 3
                        # In quadratic regression mode (so not griddata), many more points are needed
                        else:
                            objmeshpoints, NOMPoints = SamplePointsOnIntersection(self.modelPD_vtkUpSampled,
                                                                                  sphereCenter,
                                                                                  self.R,
                                                                                  NFMPoints,
                                                                                  self.griddata_mode,
                                                                                  self.C
                                                                                  )
                        # except 
                        self.objmeshpoints = objmeshpoints
                        
                    else:
                        NOMPoints = 0
                elif self.useObjectInfo:
                    # Determine the closest point to add as constraint
                    NOMPoints = 1
                    
                else:
                    NOMPoints = 0
                
                
            # Number of fluidmesh points inside the sphere
            self.info[c_idx, 0] = NFMPoints
            # Number of objectmesh points inside the sphere
            self.info[c_idx, 1] = NOMPoints
            
            if self.useObjectInfo:
                if (NOMPoints == 0) and (not self.volume_mode):
                    # print(f'No valid object mesh nodes found in sphere around point {p_idx}')
                    continue
            
# =============================================================================
#             # EE. Construct the velocity field inside the sphere
#             #         This entails:
#             #             - Masking the fluidmesh nodes
#             #             - Adding the zero-velocity nodes on the object
#             #                 (if needed)
# =============================================================================
            
            if not self.useObjectInfo:
                if NFMPoints < 4:
                    continue
                # if self.velMethod == '0' or findBL:
                    # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                # else:
                    # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                    # # 1. Apply shift
                    # velfield_in_sphere[:,:3] -= shiftVectorLocal
                    # # 2. Mask all points below 0.
                    # velfield_in_sphere = velfield_in_sphere[velfield_in_sphere[:,2] >= 0., :]
                    # # 3. Rotate the true velfield as needed
                    # velfield_in_sphere[:,:3] = funcRotateForward(velfield_in_sphere[:,:3] - vtk_point)
                
                if hasattr(self, 'vorticityNameX') and (self.velMethod=='0' or findBL):
                    velfield_in_spherePD = velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :]
                
                # velfield_in_sphere[:,:3] -= vtk_point
                self.velfieldColumn = velfield_in_sphere
                # Construct the interpolator beforehand, so we only have to do it once
                if test:
                    tCreateInterp0 = time.time()
                
                try:
                    gridDataInterp = self._estimate_velocity_at_sphere_center_with_griddata(velfield_in_sphere[:,:3],
                                                                                            velfield_in_sphere[:,3:6],
                                                                                            special=special, returnInterp=True)
                except sp.spatial.qhull.QhullError as err:
                    if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                        warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                    # Skip this point
                    continue

                if test:
                    print(f'\tIt took {time.time()-tCreateInterp0:.3f} seconds to compute interpolator')
            elif self.useSphere:
                if self.LSR_constrained:
                    if findBL:
                        # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                        if hasattr(self, 'vorticityNameX'):
                            velfield_in_spherePD = velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :]
                    else:
                        # # 0. Use all velfield nodes inside sphere
                        # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                        # # 1. Apply shift
                        # velfield_in_sphere[:,:3] -= shiftVectorLocal
                        # # 2. Mask all points below 0.
                        # velfield_in_sphere = velfield_in_sphere[velfield_in_sphere[:,2] >= 0., :]
                        # # 3. Rotate the true velfield as needed
                        # velfield_in_sphere[:,:3] = funcRotateForward(velfield_in_sphere[:,:3] - vtk_point)
                        if hasattr(self, 'vorticityNameX'):
                            # 0. Use all velfield nodes inside sphere
                            velfield_in_spherePD = velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :]
                            # 1. Apply shift
                            velfield_in_spherePD.iloc[:,:3] -= shiftVectorLocal
                            # 2. Mask all  points below 0.
                            velfield_in_spherePD.iloc[:,:3] = velfield_in_spherePD.loc[velfield_in_spherePD.iloc[:,2] >= 0., :]
                            # 3. Rotate the true velfield as needed
                            velfield_in_spherePD.iloc[:,:3] = funcRotateForward(velfield_in_spherePD.iloc[:,:3] - vtk_point)
                    
                else:

                    # self.velfield_in_sphere = velfield_in_sphere 
                    # self.objmeshpoints = objmeshpoints
                    # self.NOMPoints = NOMPoints
                    # Swap the vstack with a pre-memory allocation
                    velfield_in_sphereUse = np.vstack((velfield_in_sphere,
                                                       np.hstack((funcRotateForward(objmeshpoints - vtk_point),
                                                                  np.zeros(( int(NOMPoints), int(self.NExtraVelfieldValues) ))
                                                                  ))
                                                       ))
                    
                    # velfield_in_sphereUse = np.empty((int(NFMPoints + int(NOMPoints), int(self.NExtraVelfieldValues + 3)))
                    # Populate with fluid mesh points
                    # if findBL:
                    #     velfield_in_sphere[:NFMPoints, :] = velfieldToUse[np_fluidmeshpoints_ids, :]
                    # else:
                        
                        # 0. Use all velfield nodes inside sphere
                        # velfield_in_sphereNFM = velfieldToUse[np_fluidmeshpoints_ids, :]
                        # # # 1. Apply shift
                        # velfield_in_sphereNFM[:,:3] -= shiftVectorLocal
                        # # 2. Mask all points below 0.
                        # velfield_in_sphereNFM = velfield_in_sphereNFM[velfield_in_sphereNFM[:,2] >= 0., :]
                        # # 3. Rotate the true velfield as needed
                        # velfield_in_sphereNFM[:,:3] = funcRotateForward(velfield_in_sphereNFM[:,:3] - vtk_point)
                        
                    velfield_in_sphereUse[:NFMPoints, :] = velfield_in_sphere
                    
                    # Populate with object mesh points
                    velfield_in_sphereUse[NFMPoints:, :3] = funcRotateForward(objmeshpoints - vtk_point)
                    velfield_in_sphereUse[NFMPoints:, 3:] = 0.
                    
                    # Overwrite
                    velfield_in_sphere = velfield_in_sphereUse
                    
                    if self.velMethod == '1' and hasattr(self, 'vorticityNameX') and findBL:
                        velfield_in_column = velfieldToUse[np_fluidmeshpointsColumn_ids, :]
                        
                        velfield_in_columnPD = pd.DataFrame(velfield_in_column,
                                                            index=None,
                                                            columns=self.velfieldPD.columns
                                                            )
                    
                    if hasattr(self, 'vorticityNameX'):
                        zerosDF = pd.DataFrame(np.c_[funcRotateForward(objmeshpoints - vtk_point - shiftVectorLocal),
                                                     np.zeros((NOMPoints, self.NExtraVelfieldValues))],
                                               index=None,
                                               columns=self.velfieldPD.columns
                                               )
                        if findBL:
                            velfield_in_spherePD = pd.concat([velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :], zerosDF],
                                                                 ignore_index=True)
                        else:
                            velfieldToUsePDlocal = pd.DataFrame(velfield_in_sphere,
                                                                index=None,
                                                                columns=self.velfieldPD.columns
                                                               )
                            velfield_in_spherePD = pd.concat([velfieldToUsePDlocal, zerosDF],
                                                             ignore_index=True)
                
            self.velfield_in_sphere = velfield_in_sphere
            
        
# =============================================================================
#             # FF. Estimate velocity, velocity gradient and delta*, theta
#             #         and shape factor at the vtk_point using Least
#             #         squares or griddata
# =============================================================================
            
            if self.useCoin:
                # Use the coin to compute the velocity in each coin with a linearised mean
                
                
                if not self.coinFittingMethod=='LIN':
                    velocityAlongNormal = np.zeros((NCoins,3))
                    for coini in range(NCoins):
                        velfieldInCoin = velfieldToUse[FluidMeshPointIdsInCoin[coini], :]
                        # velfieldValuesInCoin = funcRotateForward(velfieldInCoin[:,3:6])
                        velfieldValuesInCoin = velfieldInCoin[:,3:6]
                        NpInCoin = np.ma.size(velfieldInCoin, axis=0)
                        centerCoin = dzCoin * coini
                        
                        # Estimate mean u-velocity
                        velocityAlongNormal[coini, 0], _ = self._estimateLinearMeanInCoin(centerCoin,
                                                                                                       velfieldInCoin[:,:3],
                                                                                                       velfieldValuesInCoin[:,0],
                                                                                                       NpInCoin)
                        # Estimate mean v-velocity
                        velocityAlongNormal[coini, 1], _ = self._estimateLinearMeanInCoin(centerCoin,
                                                                                                       velfieldInCoin[:,:3],
                                                                                                       velfieldValuesInCoin[:,1],
                                                                                                       NpInCoin)
                        # Estimate mean w-velocity
                        velocityAlongNormal[coini, 2], _ = self._estimateLinearMeanInCoin(centerCoin,
                                                                                                                  velfieldInCoin[:,:3],
                                                                                                                  velfieldValuesInCoin[:,2],
                                                                                                                  NpInCoin)
                
                else:
                    velocityAlongNormal = np.zeros((1,3))
                    dVel_dStep = np.zeros((3))
                    velfieldInCoin = velfield_in_sphere#velfieldToUse[np_fluidmeshpoints_ids, :]
                    # Estimate mean u-velocity
                    velocityAlongNormal[0, 0], _, dVel_dStep[0] = self._estimateLinearMeanInCoin(None,
                                                                                      velfieldInCoin[:,:3],
                                                                                      velfieldInCoin[:,3],
                                                                                      NFMPoints,
                                                                                      returnGradient=True)
                    # Estimate mean v-velocity
                    velocityAlongNormal[0, 1], _, dVel_dStep[1] = self._estimateLinearMeanInCoin(None,
                                                                                      velfieldInCoin[:,:3],
                                                                                      velfieldInCoin[:,4],
                                                                                      NFMPoints,
                                                                                      returnGradient=True)
                    # Estimate mean w-velocity
                    velocityAlongNormal[0, 2], _, dVel_dStep[2] = self._estimateLinearMeanInCoin(None,
                                                                                      velfieldInCoin[:,:3],
                                                                                      velfieldInCoin[:,5],
                                                                                      NFMPoints,
                                                                                      returnGradient=True)
                   
                # nrange = np.array([0, dzCoin])
                
                if self.coinFittingMethod=='LIN':
                    vel = velocityAlongNormal[0, :]
                    self.u[c_idx], self.v[c_idx], self.w[c_idx] = vel
                    # dVel_dStep = (velocityAlongNormal[1, :] - velocityAlongNormal[0, :]) / dzCoin
                    velWallNormalGradient = dVel_dStep * 1e3
                    self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                              (cellNormal).reshape((1,3)))
                
                # Fit a user-specified function to the mean velocities
                # Option: No-fitting, just take it directly from the linearised mean
                elif self.coinFittingMethod == 'NONE':
                    self.vel = velocityAlongNormal[int(self.useObjectInfo), :]
                    
                
                # Option: Fit a linear function through it all
                else:
                    vel = np.zeros((3,))
                    dVel_dStep = np.zeros((3,))
                    centerCoin = cylinderHeight / 2
                    velfieldInCylinder = velfieldToUse[np_fluidmeshpoints_ids, :]
                    # Estimate mean u-velocity
                    vel[0], dVel_dStep[0], _ = self._estimateLinearMeanInCylinder(np.array([0., 0., 0.,]),
                                                                                  velfieldInCylinder[:,:3],
                                                                                  velfieldInCylinder[:,3],
                                                                                  NFMPoints,
                                                                                  projectedPointTranslated = np.array([0., 0., 0.])
                                                                                  )
                    
                    # Estimate mean v-velocity
                    vel[1], dVel_dStep[1], _ = self._estimateLinearMeanInCylinder(np.array([0., 0., 0.,]),
                                                                                  velfieldInCylinder[:,:3],
                                                                                  velfieldInCylinder[:,4],
                                                                                  NFMPoints,
                                                                                  projectedPointTranslated = np.array([0., 0., 0.])
                                                                                  )
                    
                    # Estimate mean w-velocity
                    vel[2], dVel_dStep[2], _ = self._estimateLinearMeanInCylinder(np.array([0., 0., 0.,]),
                                                                                  velfieldInCylinder [:,:3],
                                                                                  velfieldInCylinder[:,5],
                                                                                  NFMPoints,
                                                                                  projectedPointTranslated = np.array([0., 0., 0.])
                                                                                  )
                    self.vel = vel
                    self.u[c_idx], self.v[c_idx], self.w[c_idx] = vel
                    velWallNormalGradient = dVel_dStep * 1e3
                    self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                              (cellNormal).reshape((1,3)))

                    
                self.isValid[c_idx] = True
                
                
            elif self.griddata_mode:
                if ((len(velfield_in_sphere[:,0] ) >= 4) and (not self.volume_mode)):
                    if not self.useObjectInfo:
                        # Can very likely not compute velocity, but we can try and extrapolate velocity to approximate this at the cell location
                        # then we can compute also tau, shape-factor and the rest
                        # We need to slowly move along the cell normal to find the first valid point to then extrapolate this towards the outputmesh
                        pointIsInvalid = True
                        pointAlongNormal = np.array([0., 0., 0.])
                        stepCount = 0
                        while pointIsInvalid and (stepCount < maxNStepsAlongNormal):
                            # Shift the velocity field (this is as if we are moving along the normal)
                                
                            # Attempt to compute velocity
                            try:
                                # Execute function
                                velRotated = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                    velfield_in_sphere[:,3:6],
                                                                                                    griddata_interp=gridDataInterp,
                                                                                                    evaluationPoint=np.c_[pointAlongNormal].T,
                                                                                                    special=special)
                                
                            except sp.spatial.qhull.QhullError as err:
                                if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                    warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                                # Mark the velocity invalid
                                velRotated = (np.nan, np.nan, np.nan)
                                                

                                
                            if ~np.isnan(velRotated[0]):
                                pointIsInvalid = False
                                self.isValid[c_idx] = True
                                
                                ###########
                                ## If velocity is valid, we compute omega
                                if hasattr(self, 'vorticityNameX'):
                                    try:
                                        omega = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                       velfield_in_spherePD.loc[:,self.omegaColumns].to_numpy(),
                                                                                                       griddata_interp=gridDataInterp,
                                                                                                       evaluationPoint=np.c_[pointAlongNormal].T,
                                                                                                       special=special)

                                    except sp.spatial.qhull.QhullError as err:
                                        if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                            warnings.warn(f'Qhull error occurred for {c_idx = } with error: "{err}"')
                                            break
                                        omega = (np.nan, np.nan, np.nan)
                                                            
                                # Get global point
                                globalPointAlongNormal = (funcRotateBackward(pointAlongNormal) + vtk_point).flatten()
                                if test:
                                    print(f'\tFirst valid velocity-point found at {globalPointAlongNormal}.')
                            else:
                                ### If function fails, then velocity is invalid and we try again
                                # Update the pointAlongNormal
                                pointAlongNormal += stepSizeAlongNormal * upVector

                                # Update the step count
                                stepCount += 1
                        
                        if pointIsInvalid:
                            ### If point is invalid we stop and continue to the next point
                            continue
                        else:
                            ### If point is valid, we compute the dynamic parameters
                            # Compute the velocity one step further away
                            pointAlongNormalNext = pointAlongNormal + stepSizeAlongNormal * upVector * 1
                                
                            velRotatedNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                    velfield_in_sphere[:,3:6],
                                                                                                    griddata_interp=gridDataInterp,
                                                                                                    evaluationPoint=np.c_[pointAlongNormalNext].T,
                                                                                                    special=special)
                            if hasattr(self, 'vorticityNameX'):
                                omegaNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                    velfield_in_spherePD.loc[:,self.omegaColumns].to_numpy(),
                                                                                                    griddata_interp=gridDataInterp,
                                                                                                    evaluationPoint=np.c_[pointAlongNormal].T,
                                                                                                    special=special)

                            #a1. Compute velocity by extrapolation
                            dvelRotated = np.array(velRotatedNext) - np.array(velRotated)
                            dVel_dStep = dvelRotated / stepSizeAlongNormal
                            deltaStepAlongNormal = stepCount * stepSizeAlongNormal
                            velocityAtVTKPoint = np.array(velRotated) - deltaStepAlongNormal * dVel_dStep
                            vel = velocityAtVTKPoint
                            
                            #a2. Compute vorticity by extrapolation
                            if hasattr(self, 'vorticityNameX'):
                                dOmega = np.array(omegaNext) - np.array(omega)
                                dOmega_dStep = dOmega / stepSizeAlongNormal
                                omegaAtVTKPoint = np.array(omega) - deltaStepAlongNormal * dOmega_dStep
                            
                            #b. Compute the in-plane velocity wall-normal gradient components
                            velWallNormalGradient = dVel_dStep * 1e3
                            self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                                      (cellNormal).reshape((1,3)))
                            
                            
                            #c. Compute the delta* and theta parameters
                            #i. Compute the normal velocity profile until the approximated boundary layer height
                            # Define flag for boundary layer height, based on vorticity
                            if hasattr(self, 'vorticityNameX') and findBL:
                                vorticityNotConverged = True
                                stepCountVorticity = stepCount
                                initialStepFactor = 10
                                useLargeStep = True
                                pointAlongNormal = np.array([0., 0., 1.e-15])
                                if test:
                                    print('Iterating to find boundary layer height')
                                while vorticityNotConverged and (stepCountVorticity < maxNStepsAlongNormal):
                                    #### Condition 1: Based on a threshold value of the total vorticity
                                    pointAlongNormalProfile = pointAlongNormal + stepSizeAlongNormal * upVector * (stepCountVorticity+1)
                                    
                                    # # Compute vorticity at the point using vtk's gradient filter
                                    omegaNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                       velfield_in_spherePD.loc[:,self.omegaColumns].to_numpy(),
                                                                                                       griddata_interp=gridDataInterp,
                                                                                                       evaluationPoint=np.c_[pointAlongNormalProfile].T,
                                                                                                       special=special)
                                    omegaMag = np.linalg.norm(omegaNext)
                                    
                                    if test:
                                        print(f'Computed vorticity at point ({np.round(pointAlongNormalProfile, 2)}): omegaMag = {np.round(omegaMag, 2)}')
                                    
                                    if omegaMag < omegaThreshold:
                                        if useLargeStep and (stepCountVorticity >= initialStepFactor):
                                            useLargeStep = False
                                            stepCountVorticity += (1 - initialStepFactor)
                                            continue
                                        # Converged
                                        vorticityNotConverged = False
                                        # Save the results
                                        delta = (stepCountVorticity + 1) * stepSizeAlongNormal
                                        self.delta[c_idx] = delta
                                        self.BLvalid[c_idx] = True
                                        if test:
                                            print(f'\tHeight of boundary layer estimated to be {delta:.1f}mm')
                                        # Compute delta* and theta using the integral
                                        
                                        # and lastly the shape Factor
                                        
                                    else:
                                        # stepCount based on large step or small step
                                        if useLargeStep:
                                            stepCountVorticity += initialStepFactor
                                        else:
                                            stepCountVorticity += 1
                                if self.delta[c_idx] == 0:
                                    self.BLvalid[c_idx] = False
                                    if test:
                                        print('\tBoundary layer iterate not converged.')

                    else:
                        # Compute the cell-centered velocity using griddata
                        
                        # 0. Build the interpolators
                        try:
                            gridDataInterp = self._estimate_velocity_at_sphere_center_with_griddata(velfield_in_sphere[:,:3],
                                                                                                    velfield_in_sphere[:,3:6],
                                                                                                    special=special, returnInterp=True)
                            if findBL:
                                gridDataInterpColumn = self._estimate_velocity_at_sphere_center_with_griddata(velfield_in_column[:,:3],
                                                                                                              velfield_in_column[:,3:6],
                                                                                                              special=special, returnInterp=True)
                        except sp.spatial.qhull.QhullError as err:
                            if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                            # Skip this point
                            continue
                        
                        #1. Velocity at point
                        try:
                            vel = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                         velfield_in_sphere[:,3:6],
                                                                                         griddata_interp=gridDataInterp,
                                                                                         evaluationPoint=np.array([[0, 0, 1e-15]]),
                                                                                         special=special)
                            
                            omegaAtVTKPoint = np.zeros((3,))
                            
                        except sp.spatial.qhull.QhullError as err:
                            if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                                continue
                            vel = (np.nan, np.nan, np.nan)
                            
                        ###2. Velocity gradient
                        #a. We compute the velocity at point one step away from the current point
                        pointNext = stepSizeAlongNormal * upVector
                        try:
                            velNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                         velfield_in_sphere[:,3:6],
                                                                                         griddata_interp=gridDataInterp,
                                                                                         evaluationPoint=np.c_[pointNext].T,
                                                                                         special=special)
                        except sp.spatial.qhull.QhullError as err:
                            if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                                continue
                        
                        #b. Compute the gradient using a forward difference scheme
                        dvel = np.array(velNext) - np.array(vel)
                        dVel_dStep = dvel / stepSizeAlongNormal
                        self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((dVel_dStep).reshape((1,3)) * 1e3,
                                                                                  (cellNormal).reshape((1,3)))
                        
                        #3. Boundary layer height
                        #i. Compute the normal velocity profile until the approximated boundary layer height
                        # Define flag for boundary layer height, based on vorticity
                        if hasattr(self, 'vorticityNameX') and findBL:
                            vorticityNotConverged = True
                            stepCountVorticity = 0
                            initialStepFactor = 10
                            useLargeStep = True
                            pointAlongNormal = np.array([0., 0., 1.e-15])
                            # maxNStepsAlongNormal = self.R / stepSizeAlongNormal
                            if test:
                                print('Iterating to find boundary layer height')
                            while vorticityNotConverged and (stepCountVorticity < maxNStepsAlongNormal):
                                #### Condition 1: Based on a threshold value of the total vorticity
                                pointAlongNormalProfile = pointAlongNormal + stepSizeAlongNormal * upVector * (stepCountVorticity+1)
    
                                # # Compute vorticity at the point using griddata
                                try:
                                    omegaNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                       velfield_in_columnPD.loc[:,self.omegaColumns].to_numpy(),
                                                                                                       griddata_interp=gridDataInterpColumn,
                                                                                                       evaluationPoint=np.c_[pointAlongNormalProfile].T,
                                                                                                       special=special)
                                except sp.spatial.qhull.QhullError as err:
                                    if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                        warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                                        
                                    omegaNext = (np.nan, np.nan, np.nan)
                                omegaMag = np.linalg.norm(omegaNext)
                                
                                if test:
                                    print(f'Computed vorticity at point ({np.round(pointAlongNormalProfile, 2)}): omegaMag = {np.round(omegaMag, 2)}')
                                
                                if omegaMag < omegaThreshold:
                                    if useLargeStep and (stepCountVorticity >= initialStepFactor):
                                        useLargeStep = False
                                        stepCountVorticity += (1 - initialStepFactor)
                                        continue
                                    # Converged
                                    vorticityNotConverged = False
                                    # Save the results
                                    delta = (stepCountVorticity + 1) * stepSizeAlongNormal
                                    self.BLvalid[c_idx] = True
                                    self.delta[c_idx] = delta
                                    if test:
                                        print(f'\tHeight of boundary layer estimated to be {delta:.1f}mm')
                                    # Compute delta* and theta using the integral
                                    
                                    # and lastly the shape Factor
                                    
                                else:
                                    # stepCount based on large step or small step
                                    if useLargeStep:
                                        stepCountVorticity += initialStepFactor
                                    else:
                                        stepCountVorticity += 1
                            if self.delta[c_idx] == 0:
                                self.BLvalid[c_idx] = False
                                if test:
                                    print('\tBoundary layer iterate not converged.')
                        
                    
                    # Assign the velocity
                    self.u[c_idx], self.v[c_idx], self.w[c_idx] = vel
                    
                    # Assign the vorticity
                    if hasattr(self, 'vorticityNameX'):
                        self.omegaX[c_idx], self.omegaY[c_idx], self.omegaZ[c_idx] = omegaAtVTKPoint

                    # Determine if the velocity is valid
                    self.isValid[c_idx] = ~np.isnan(vel[0])
                else:
                    continue
            else:
                # Do not compute LSR if number of particles is less than 10.
                # This is not enough for LSR with 3*10 coefficients.
                if self.info[c_idx, 0] < 10:
                    continue
                
                if div_free:
                    try:
                        vel, res = self._estimate_velocity_at_sphere_center_with_LSR_divfree(velfield_in_sphere[:,:3],
                                                                                             velfield_in_sphere[:,3:6],
                                                                                             order=self.LSR_ORDER)
                    except np.linalg.LinAlgError as err:
                        if 'Singular matrix' in str(err):
                            continue
                        else:
                            raise NotImplementedError(f'Different error occurred:\n{str(err)}')

                    self.u[c_idx], self.v[c_idx], self.w[c_idx] = vel
                    res_u, res_v, res_w = res
                else:
                    
                    ####### Here we execute method 4 #######
                    if self.LSR_constrained:
                        # Check if sphere intersects ground
                        dist2ground = vtk_point[2]
                        if dist2ground <= self.R:
                        # if (vtk_point[2] - self.R) <= 0.: #sphere intersects ground
                            sphere_intersects_ground = True
                        else:
                            sphere_intersects_ground = False
                            
                        # Check if sphere intersects with object itself
                        closest_point = np.array([0.0, 0.0, 0.0])
                        projected_cellId_obj = vtk.reference(0)
                        subId_obj = vtk.reference(0)
                        dist2_obj = vtk.reference(0.)
                        cell_obj = vtk.vtkGenericCell()
                        sphere_intersects_model = vtk.reference(0)
                        
                        self.cellLocator_objectSTL.FindClosestPointWithinRadius(vtk_point,
                                                                                self.R,
                                                                                closest_point,
                                                                                cell_obj,
                                                                                projected_cellId_obj,
                                                                                subId_obj,
                                                                                dist2_obj,
                                                                                sphere_intersects_model)
                        dist_obj = np.sqrt(dist2_obj.get())
                        
                        # Combine checks to see if wall constraint needs to be applied
                        sphere_intersects_surface = sphere_intersects_ground or sphere_intersects_model
                        Use_sphere_intersects_ground = (dist2ground < dist_obj) and sphere_intersects_ground
                        Use_sphere_intersects_model = (dist2ground >= dist_obj) and sphere_intersects_model
                    else:
                        sphere_intersects_ground = False
                        Use_sphere_intersects_ground = False
                        Use_sphere_intersects_model = False
                        
                        
                    if (self.LSR_constrained) and sphere_intersects_surface:
                        ## Estimate velocity using constrained optimization
                        # 1a. Get the projected point (check for ground intersection)
                        if Use_sphere_intersects_ground:
                            projected_point = np.array([vtk_point[0], vtk_point[1], 0]) - vtk_point
                            # self.projected_point_ground = projected_point_ground
                            
                        # 1b. Get the projected point (check for model intersection)
                        if Use_sphere_intersects_model:
                            # 1. Get the projected point and cell in the objectmesh
                            projected_point = closest_point.astype(np.float32) - vtk_point
                        
                        # Execute constrained Least-Squares
                        solution = pivOpt.optimizeLSR_wCons(velfield_in_sphere[:,:3],
                                                            velfield_in_sphere[:,3:6],
                                                            projected_point,
                                                            use_all_constraints=False
                                                            )

                        # Compute the velocity at the center point
                        beta = solution['x']
                        self.beta = beta
                        self.u[c_idx], self.v[c_idx], self.w[c_idx] = beta[[0,10,20]].flatten()
                        
                        # Extract the derivatives with respect to x at the center point
                        self.dudx[c_idx], self.dvdx[c_idx], self.dwdx[c_idx] = funcRotateBackward(np.c_[beta[[1,11,21]]].T).flatten()
                        # Extract the derivatives with respect to x at the center point
                        self.dudy[c_idx], self.dvdy[c_idx], self.dwdy[c_idx] = funcRotateBackward(np.c_[beta[[2,12,22]]].T).flatten()
                        # Extract the derivatives with respect to x at the center point
                        self.dudz[c_idx], self.dvdz[c_idx], self.dwdz[c_idx] = funcRotateBackward(np.c_[beta[[3,13,23]]].T).flatten()
                        
                        # Compute the vorticity and store
                        omega_x = self.dwdy[c_idx] - self.dvdz[c_idx]
                        omega_y = self.dudz[c_idx] - self.dwdx[c_idx]
                        omega_z = self.dvdx[c_idx] - self.dudy[c_idx]
                        self.omega[c_idx, :] = np.array([omega_x, omega_y, omega_z])
                        
                        # Store the wall-normal gradients
                        dVel_dStep = np.array([beta[3], beta[13], beta[23]])
                        velWallNormalGradient = dVel_dStep * 1e3
                        self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                                  (cellNormal).reshape((1,3)))
                        
                        # # Compute residuals in u
                        # res_u = velfield_in_sphere[:, 3] - pivOpt.vel3D_func_fast(beta[:10], velfield_in_sphere[:, :3]).T
                        # # In v
                        # res_v = velfield_in_sphere[:, 4] - pivOpt.vel3D_func_fast(beta[10:20], velfield_in_sphere[:, :3]).T
                        # # And in w
                        # res_w = velfield_in_sphere[:, 5] - pivOpt.vel3D_func_fast(beta[20:30], velfield_in_sphere[:, :3]).T

                    else:
                        ####### Here we execute methods 2 and 3 ####### (Omg so much bs before this.....)
                        # estimate velocity-u
                        (self.u[c_idx], 
                         beta_u, res_u) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphere[:,:3],
                                                                                            velfield_in_sphere[:,3],
                                                                                            order=self.LSR_ORDER)
                        # estimate velocity-v
                        (self.v[c_idx],
                         beta_v, res_v) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphere[:,:3],
                                                                                            velfield_in_sphere[:,4],
                                                                                            order=self.LSR_ORDER)
                        # estimate velocity-w
                        (self.w[c_idx],
                         beta_w, res_w) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphere[:,:3],
                                                                                            velfield_in_sphere[:,5],
                                                                                            order=self.LSR_ORDER)
                        
                        # Get the gradient coefficients from beta
                        self.dudx[c_idx], self.dudy[c_idx], self.dudz[c_idx] = funcRotateBackward(np.c_[beta_u[1:4]].T).flatten() * 1e3
                        self.dvdx[c_idx], self.dvdy[c_idx], self.dvdz[c_idx] = funcRotateBackward(np.c_[beta_v[1:4]].T).flatten() * 1e3
                        self.dwdx[c_idx], self.dwdy[c_idx], self.dwdz[c_idx] = funcRotateBackward(np.c_[beta_w[1:4]].T).flatten() * 1e3
                        
                        # Compute the vorticity and store
                        omega_x = self.dwdy[c_idx] - self.dvdz[c_idx]
                        omega_y = self.dudz[c_idx] - self.dwdx[c_idx]
                        omega_z = self.dvdx[c_idx] - self.dudy[c_idx]
                        self.omega[c_idx, :] = np.array([omega_x, omega_y, omega_z])
                        
                        # Store coefficients
                        self.beta = np.r_[beta_u, beta_v, beta_w]
                        
                        # Store the wall-normal gradients
                        dVel_dStep = np.array([beta_u[3], beta_v[3], beta_w[3]])
                        velWallNormalGradient = dVel_dStep * 1e3
                        self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                                  (cellNormal).reshape((1,3)))
                        
                        # Now walk along the normal and compute the vorticity to threshold
                        if hasattr(self, 'vorticityNameX') and findBL:
                            vorticityNotConverged = True
                            stepCountVorticity = 0
                            initialStepFactor = 10
                            useLargeStep = True
                            pointAlongNormal = np.array([0., 0., 0])
                            if test:
                                print('Iterating to find boundary layer height')
                            while vorticityNotConverged and (stepCountVorticity < maxNStepsAlongNormal):
                                #### Condition 1: Based on a threshold value of the total vorticity
                                pointAlongNormalProfile = pointAlongNormal + stepSizeAlongNormal * upVector * (stepCountVorticity+1)
    
                                ###############################################
                                ########### A. Determine the fluidmesh ids
                                ###############################################
                                # 1. Determine velfield ids
                                np_fluidmeshpoints_idsNext = self._find_ids_in_sphere_with_Locator(pointLocator_fluidmeshToUse,
                                                                                                   self.R,
                                                                                                   pointAlongNormalProfile)
                                NFMPointsNext = np.size(np_fluidmeshpoints_idsNext)
                                
                                ###############################################
                                ########### B. Determine the objectmesh points
                                ###############################################
                                # Find the section of the circle which intersects
                                # 1. Intersects with the ground
                                ground_intersect = self._find_ground_intersect(pointAlongNormalProfile,
                                                                               self.R,
                                                                               z_ground=0.)
                                
                                # 2. Intersects with the model
                                # Check if sphere intersects with object itself
                                closest_point = np.array([0.0, 0.0, 0.0])
                                projected_cellId_obj = vtk.reference(0)
                                subId_obj = vtk.reference(0)
                                dist2_obj = vtk.reference(0.)
                                cell_obj = vtk.vtkGenericCell()
                                model_intersect = vtk.reference(0)
                                
                                self.cellLocator_objectSTL.FindClosestPointWithinRadius(pointAlongNormalProfile,
                                                                                        self.R,
                                                                                        closest_point,
                                                                                        cell_obj,
                                                                                        projected_cellId_obj,
                                                                                        subId_obj,
                                                                                        dist2_obj,
                                                                                        model_intersect
                                                                                        )
                                
                                # Check if closer to ground or model
                                closerToGround = (not model_intersect) or (pointAlongNormalProfile[2] <= np.sqrt(dist2_obj.get()))
                                
                                if (ground_intersect is not None) and closerToGround:
                                    Vs = self._VolumeCutSphere(self.R, self.R - pointAlongNormalProfile[2])  
                                    
                                    NOMPointsNext = self._NOMPoints_from_NFMPoints(NFMPointsNext, ground_intersect[1])
                                    # round(NFMPointsNext  * ground_intersect[1] / (Vs *3 / (4*np.pi))**(1/3))
                                    
                                    # Using a poisson sample distribution,
                                    # place N points inside the circle of intersection
                                    groundCenter = np.array([pointAlongNormalProfile[0],
                                                             pointAlongNormalProfile[1],
                                                             0.])
                                    groundCircleRadius = self.R
                                    objmeshpointsNext = self._poissonSampleDistributionCircle(groundCenter,
                                                                                              groundCircleRadius,
                                                                                              NOMPointsNext)

                                elif model_intersect:
                                    (objmeshpointsNext,
                                     NOMPointsNext) = SamplePointsOnIntersection(self.modelPD_vtkUpSampled,
                                                                                 pointAlongNormalProfile,
                                                                                 self.R,
                                                                                 NFMPointsNext ,
                                                                                 self.griddata_mode,
                                                                                 self.C
                                                                                 )

                                # Combine all together
                                velfield_in_sphereNext = np.vstack((velfieldToUse[np_fluidmeshpoints_idsNext, :], 
                                                                    np.c_[funcRotateForward(objmeshpointsNext),
                                                                          np.zeros((NOMPointsNext, self.NExtraVelfieldValues))] ))
                                velfield_in_sphereNext[:,:3] -= pointAlongNormalProfile
    
                                # # Compute vorticity at the point using griddata
                                (_, beta_uNext, _) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphereNext[:,:3],
                                                                                                       velfield_in_sphereNext[:,3],
                                                                                                       order=self.LSR_ORDER)
                                # estimate velocity-v
                                (_, beta_vNext, _) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphereNext[:,:3],
                                                                                                       velfield_in_sphereNext[:,4],
                                                                                                       order=self.LSR_ORDER)
                                # estimate velocity-w
                                (_, beta_wNext, _) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphereNext[:,:3],
                                                                                                       velfield_in_sphereNext[:,5],
                                                                                                       order=self.LSR_ORDER)
                                
                                # Get the gradient coefficients from beta
                                _,    dudy, dudz = funcRotateBackward(np.c_[beta_uNext[1:4]].T).flatten() * 1e3
                                dvdx, _,    dvdz = funcRotateBackward(np.c_[beta_vNext[1:4]].T).flatten() * 1e3
                                dwdx, dwdy, _    = funcRotateBackward(np.c_[beta_wNext[1:4]].T).flatten() * 1e3
                                
                                # Compute the vorticity and store
                                omega_x = dwdy - dvdz
                                omega_y = dudz - dwdx
                                omega_z = dvdx - dudy
                                omegaNext = np.array([omega_x, omega_y, omega_z])
                                
                                omegaMag = np.linalg.norm(omegaNext)
                                
                                if test:
                                    print(f'Computed vorticity at point ({np.round(pointAlongNormalProfile, 2)}): omegaMag = {np.round(omegaMag, 2)}')
                                
                                if omegaMag < omegaThreshold:
                                    if useLargeStep and (stepCountVorticity >= initialStepFactor):
                                        useLargeStep = False
                                        stepCountVorticity += (1 - initialStepFactor)
                                        continue
                                    # Converged
                                    vorticityNotConverged = False
                                    # Save the results
                                    delta = (stepCountVorticity + 1) * stepSizeAlongNormal
                                    self.BLvalid[c_idx] = True
                                    self.delta[c_idx] = delta
                                    if test:
                                        print(f'\tHeight of boundary layer estimated to be {delta:.1f}mm')
                                    # Compute delta* and theta using the integral
                                    
                                    # and lastly the shape Factor
                                    
                                else:
                                    # stepCount based on large step or small step
                                    if useLargeStep:
                                        stepCountVorticity += initialStepFactor
                                    else:
                                        stepCountVorticity += 1
                            if self.delta[c_idx] == 0:
                                self.BLvalid[c_idx] = False
                                if test:
                                    print('\tBoundary layer iterate not converged.')
                # Store valid array
                self.isValid[c_idx] = True
            
            # 5. Save information regarding the process
            # RMS of LSR residual for velocity u
            if (not self.griddata_mode) and (not self.LSR_constrained) and self.useSphere:
                self.info[c_idx, 2] = self._RMS(res_u)
                # RMS of LSR residual for velocity v
                self.info[c_idx, 3] = self._RMS(res_v)
                # RMS of LSR residual for velocity w
                self.info[c_idx, 4] = self._RMS(res_w)
                # if (not self.volume_mode) and (hasattr(self, 'objectmesh_vtk')):
                if self.useObjectInfo and (not self.LSR_constrained):
                    # RMS error at object mesh points specifically for velocity u
                    # self.info[p_idx, 5] = self._RMS(res_u[-len(np_objmeshpoints_ids):])
                    self.info[c_idx, 5] = self._RMS(res_u[-NOMPoints:])
                    # RMS error at object mesh points specifically for velocity v
                    # self.info[p_idx, 6] = self._RMS(res_v[-len(np_objmeshpoints_ids):])
                    self.info[c_idx, 6] = self._RMS(res_v[-NOMPoints:])
                    # RMS error at object mesh points specifically for velocity w
                    # self.info[p_idx, 7] = self._RMS(res_w[-len(np_objmeshpoints_ids):])
                    self.info[c_idx, 7] = self._RMS(res_w[-NOMPoints:])
                # RSS of LSR regression for velocity u
                self.info[c_idx, 8] = self._RSS(res_u)
                # RSS of LSR regression for velocity v
                self.info[c_idx, 9] = self._RSS(res_v)
                # RSS of LSR regression for velocity w
                self.info[c_idx, 10] = self._RSS(res_w)
                
                # Compute RSS of fluctuations (only consider fluid mesh points)
                self.info[c_idx, 11] = np.mean(res_u[:NFMPoints] * res_u[:NFMPoints])
                
            # Measure processing time and save to array
            if test:
                if (c_idx-c_start) < N_timesteps:
                    t1 = time.time()
                    time_lst[c_idx-c_start] = t1 - t0
                else:
                    avg_time_per_step = np.mean(time_lst)
                    print(f'Average performance of {round(avg_time_per_step*1000, 2)} milliseconds per timestep\n(averaged over {N_timesteps} steps).\nTotal computation would take approx {round(avg_time_per_step * self.Ncells)} seconds')
                    break
        
        self.Ncells_valid = round(sum(self.isValid))
        print(f'Sphere-Iteration done. {self.Ncells_valid}/{self.Ncells} valid results computed.')
        print(f'On average encountered {np.round(self.avg_NFM, 2)} fluid-mesh points inside spheres.')
        pass
    
    def _plot_after_evaluation(self):
        fig = plt.figure(figsize=plt.figaspect(1/1.5))
        
        velfield_rot = self.funcRotateBackward(self.velfield_in_sphere[:,:3])
        fig.suptitle(f'Points for cell id {self.c_idx} ({np.round(self.point[0], 2)}, {np.round(self.point[1], 2)}, {np.round(self.point[2], 2)}) - velmethod = {self.velMethod} - NFMpoints = {self.NFMPoints}', fontsize=16)

        # Create the figure with subplots
        ax1 = fig.add_subplot(231)
        ax1.plot(velfield_rot[self.NFMPoints:,0] + self.point[0], velfield_rot[self.NFMPoints:,2] + self.point[2], 'r.', label='Object points')
        ax1.plot(velfield_rot[:self.NFMPoints,0] + self.point[0], velfield_rot[:self.NFMPoints,2] + self.point[2], 'c.', label='Fluid points')
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Z [mm]')
        ax1.set_xlim(self.point[0] - self.R, self.point[0] + self.R)
        ax1.set_ylim(self.point[2] - self.R, self.point[2] + self.R)
        ax1.set_title('Side view in x-z plane')
        
        ax2 = fig.add_subplot(232, sharey=ax1)
        ax2.plot(velfield_rot[self.NFMPoints:,1] + self.point[1], velfield_rot[self.NFMPoints:,2] + self.point[2], 'r.', label='Object points')
        ax2.plot(velfield_rot[:self.NFMPoints,1] + self.point[1], velfield_rot[:self.NFMPoints,2] + self.point[2], 'c.', label='Fluid points')
        ax2.set_xlabel('Y [mm]')
        ax2.invert_xaxis()
        ax2.set_ylabel('Z [mm]')
        ax2.set_xlim(self.point[1] - self.R, self.point[1] + self.R)
        ax2.set_title('Front view in y-z plane')
        
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.scatter(*(velfield_rot[self.NFMPoints:,:] + self.point).T, c = 'r', marker='.', label='Object points')
        ax3.scatter(*(velfield_rot[:self.NFMPoints,:] + self.point).T, c = 'c', marker='.', label='Fluid points')
        ax3.set_xlabel('X [mm]')
        ax3.set_ylabel('Y [mm]')
        ax3.set_zlabel('Z [mm]')
        ax3.set_xlim(self.point[0] - self.R, self.point[0] + self.R); ax3.set_ylim(self.point[1] - self.R, self.point[1] + self.R) ; ax3.set_zlim(self.point[2] - self.R, self.point[2] + self.R)
        ax3.set_title('3D view')
        
        handles_top, labels_top = ax3.get_legend_handles_labels()
        fig.legend(handles_top, labels_top, loc=(0.86, 0.7), fontsize=16)
        
        #### Add solution estimate
        lw=3
        if self.velMethod == '2' or self.velMethod == '3' or self.velMethod == '4':
            zrange = np.linspace(min(self.velfield_in_sphere[:,2].min(), 0), self.velfield_in_sphere[:,2].max(), 100)
        elif self.velMethod == 'GT':
            zrange = np.linspace(min(self.velfield_in_sphere[:,2].min(), 0), self.cylinderHeight, 100)
        self.zrange = zrange
        # yrange = np.ones_like(zrange) * self.point[1]
        yrange = np.zeros_like(zrange)
        # xrange = np.ones_like(zrange) * self.point[0]
        xrange = np.zeros_like(zrange)
        points = np.c_[xrange, yrange, zrange]
        velocity_fit = self.get_velocity(points)
        
        ax4 = fig.add_subplot(234)
        ax4.plot(self.velfield_in_sphere[:,3], self.velfield_in_sphere[:,2], 'k.', label='Data')
        if self.velMethod == '2' or self.velMethod == '3' or self.velMethod == '4':
            ax4.plot(velocity_fit[:,0], zrange, '--', color='tab:orange', label='Fit', lw=lw)
        elif self.velMethod == 'GT':
            # vel_estimate = self.vel[0] + zrange * self.normalGradient[0] / 1e3
            vel_estimate = velocity_fit[:,0]
            ax4.plot(vel_estimate, zrange, '--', color='tab:orange', label='Fit', lw=lw)
        ax4.set_xlabel('u [m/s]')
        ax4.set_ylabel('n [mm]')
        
        
        ax5 = fig.add_subplot(235)
        ax5.plot(self.velfield_in_sphere[:,4], self.velfield_in_sphere[:,2], 'k.', label='Data')
        if self.velMethod == '2' or self.velMethod == '3' or self.velMethod == '4':
            ax5.plot(velocity_fit[:,1], zrange, '--', color='tab:orange', label='Fit', lw=lw)
        elif self.velMethod == 'GT':
            # vel_estimate = self.vel[1] + zrange * self.normalGradient[1] / 1e3
            vel_estimate = velocity_fit[:,1]
            ax5.plot(vel_estimate, zrange, '--', color='tab:orange', label='Fit', lw=lw)
        ax5.set_xlabel('v [m/s]')
        ax5.set_ylabel('n [mm]')
        
        ax6 = fig.add_subplot(236)
        ax6.plot(self.velfield_in_sphere[:,5], self.velfield_in_sphere[:,2], 'k.', label='Data')
        if self.velMethod == '2' or self.velMethod == '3' or self.velMethod == '4':
            ax6.plot(velocity_fit[:,2], zrange, '--', color='tab:orange', label='Fit', lw=lw)
        elif self.velMethod == 'GT':
            # vel_estimate = self.vel[2] + zrange * self.normalGradient[2] / 1e3
            vel_estimate = velocity_fit[:,2]
            ax6.plot(vel_estimate, zrange, '--', color='tab:orange', label='Fit', lw=lw)
        ax6.set_xlabel('w [m/s]')
        ax6.set_ylabel('n [mm]')
        
        handles_bottom, labels_bottom = ax6.get_legend_handles_labels()
        fig.legend(handles_bottom, labels_bottom, loc=(0.86, 0.3), fontsize=16)
        fig.subplots_adjust(left=0.05, right=0.85)
        pass
    
    def get_velocity(self, pts):
        pts = np.asarray(pts).reshape((-1, 3))
        # Return the velocity depending on the method used.
        
        # Method 2, 3 and 4 -> Use fitted parameters, with beta
        if (self.velMethod == '2') or (self.velMethod == '3') or (self.velMethod == '4'):
            X = self._X_for_tri_polynomial(pts, order=self.LSR_ORDER)
            # X_full = np.hstack((X, X, X))
            
            vel_u = np.dot(X, self.beta[:10])
            vel_v = np.dot(X, self.beta[10:20])
            vel_w = np.dot(X, self.beta[20:])
            
            vel = np.c_[vel_u, vel_v, vel_w]
        
        # Method GT is always a linear fit (for now at least)
        elif (self.velMethod == 'GT'):
            # vel = np.zeros_like(pts)
            
            # Get the velocity at center
            vel_u = self.vel[0] + pts[:,2] * self.normalGradient[0] / 1e3
            vel_v = self.vel[1] + pts[:,2] * self.normalGradient[1] / 1e3
            vel_w = self.vel[2] + pts[:,2] * self.normalGradient[2] / 1e3
            
            vel = np.c_[vel_u, vel_v, vel_w]
            
        # Lastly is the griddata methods
        else:
            vel = np.zeros_like(pts)
            
        return vel
            
    def ShowEvaluationAtCellById(self, c_idx, track_percentage=False, div_free=False, test=False, partitions=10,
                                 special=False, use_all_constraints=False, dz_ground=0,
                                 Rvary=12, Rlocal=50, minDistanceToObject=12,
                                 stepSizeAlongNormal = 1e-1, maxNStepsAlongNormal=400,
                                 omegaThreshold=100, c_start=0, findBL=False, plotResults=True,
                                 returnResults=False):
        # We want to evaluate the method at the mentioned point
        
        # 1. Get the cell info
        vtk_cell, vtk_point, cellNormal = self._get_cell_info(c_idx)
                
        dist2ToObj, closest_point = self.FindClosestPointToObject(vtk_point)
        
        self.c_idx = c_idx
        self.point = vtk_point
        self.cell = vtk_cell
        self.normal = cellNormal
        
        # Compute distance to object
        _, dist2_obj = self.FindClosestPointToObject(vtk_point)
        distanceToObject = np.sqrt(dist2_obj)
        
        # Determine functions to rotate the whole shabang to and from
        # cellNormal using Rodrigues' rotation
        upVector = np.array([0., 0., 1.])
        (funcRotateForward,
         funcRotateBackward,
         rotationAngle,
         rotMat) = self.GetRotationInfo(cellNormal, upVector)
        
        self.funcRotateForward, self.funcRotateBackward, self.rotation = (funcRotateForward,
                                                                          funcRotateBackward,
                                                                          rotationAngle)
        
# =============================================================================
#         # 2. Apply shift and stuff
# =============================================================================
        if self.velMethod == '0' or findBL:
            # In this case we find a local velocity field first which we
            # speed up the computations (does it actually speed up...?)
            
            # The local velocity field is shifted and rotated
            
            if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
                if test:
                    print('\tLocating shift in object location for current cell')
                self._find_ground_shift_for_cell(c_idx, vtk_point,
                                                 vtk_cell, distanceToObject,
                                                 funcRotateForward, funcRotateBackward,
                                                 Rvary, dz_ground, Rlocal
                                                 )
                # Determine the shift in ground plane
                if self.groundPlaneFitComputed:
                    shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
                else:
                    shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                                  Rvary, dz_ground, Rlocal=Rlocal)
            else:
                shiftVectorLocal = np.zeros(3,)
            # Use a localized velocity field and shift the velocity field
            velfieldLocal = self._getLocalVelfield(self.pointLocator_fluidmesh, Rlocal, vtk_point, self.velfield)
            coorLocalTranslated = velfieldLocal[:,:3] - shiftVectorLocal
            uvwLocal = velfieldLocal[:,3:6]
            
            # Crop the coordinates if they fall inside the ground
            maskCoordinatesLocal = coorLocalTranslated[:,2] >= 0
            
            coorLocalTransformedCropped = funcRotateForward(coorLocalTranslated[maskCoordinatesLocal, :] - vtk_point)
            uvwLocalCropped = uvwLocal[maskCoordinatesLocal, :]
            
            # Build the translated velocity field
            (fluidmeshLocalTranslated_vtk,
             velfieldLocalTranslated) = self._buildTransformedLocalFluidMesh(coorLocalTransformedCropped,
                                                                             uvwLocalCropped,
                                                                             velfieldLocal[maskCoordinatesLocal,6:]
                                                                             )

            # Update the fluidmesh point locator
            fluidmeshToUse = fluidmeshLocalTranslated_vtk
            velfieldToUse = velfieldLocalTranslated
            self.velfieldToUse = velfieldLocalTranslated
            self.shift = shiftVectorLocal
            self.angle = rotationAngle
            applyRotationAfterwards = False
            
            # Build a pointLocator for the local fluid mesh
            pointLocator_fluidmeshToUse = self._build_vtkPointLocator(fluidmeshToUse, 100)
            
        elif self.useCoin:
            # In this case, we use a local velocity field, but do not build
            # a special locator for each cell, since we don't use that
            
            # The velocity field is subsequently shifted and rotated
            if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
                if test:
                    print('\tLocating shift in object location for current cell')
                # Determine the shift in ground plane
                if self.groundPlaneFitComputed:
                    shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
                else:
                    shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                                  Rvary, dz_ground, Rlocal=Rlocal)
                    
            else:
                shiftVectorLocal = np.zeros(3,)
                
            # # Use a localized velocity field and shift the velocity field
            # RlocalField = 5*np.sqrt((self.coinHeight * 2)**2  + (self.R)**2)
            # velfieldLocal = self._getLocalVelfield(self.pointLocator_fluidmesh,
            #                                        RlocalField,
            #                                        vtk_point,
            #                                        self.velfield)
            # coorLocalTranslated = velfieldLocal[:,:3] - shiftVectorLocal
            # uvwLocal = velfieldLocal[:,3:6]
            
            # # Crop the coordinates if they fall inside the ground
            # maskCoordinatesLocal = coorLocalTranslated[:,2] >= 0
            
            # coorLocalTransformedCropped = funcRotateForward(coorLocalTranslated[maskCoordinatesLocal, :] - vtk_point)
            # uvwLocalCropped = uvwLocal[maskCoordinatesLocal, :]
            
            # # Build the translated velocity field
            # (fluidmeshLocalTranslated_vtk,
            #  velfieldLocalTranslated) = self._buildTransformedLocalFluidMesh(coorLocalTransformedCropped,
            #                                                                  uvwLocalCropped,
            #                                                                  velfieldLocal[maskCoordinatesLocal,6:]
            #                                                                  )

            # # Update the fluidmesh point locator
            # velfieldToUse = velfieldLocalTranslated
            # fluidmeshToUse = fluidmeshLocalTranslated_vtk
            # self.velfieldToUse = velfieldLocalTranslated
            # self.shift = shiftVectorLocal
            # self.angle = rotationAngle

            fluidmeshToUse = self.fluidmesh_vtk
            velfieldToUse = self.velfield
            pointLocator_fluidmeshToUse = self.pointLocator_fluidmesh
            applyRotationAfterwards = True
        else:
            # In this case we do not change the velocity field.
            # Only the ground-shift is determined, which is later used to
            # shift the velocity field when fluidmesh nodes inside the
            # local sphere are considered
            if self.updateGroundEstimate and (distanceToObject > minDistanceToObject):
                # Determine the shift in ground plane
                if self.groundPlaneFitComputed:
                    shiftVectorLocal = self.shiftedPoints[c_idx, :] - vtk_point
                else:
                    shiftVectorLocal = self._updateGroundEstimate(vtk_cell, vtk_point, funcRotateForward, funcRotateBackward,
                                                                  Rvary, dz_ground, Rlocal=Rlocal)
                    self.shiftVector[c_idx, :] = shiftVectorLocal
            else:
                shiftVectorLocal = np.zeros(3,)

            fluidmeshToUse = self.fluidmesh_vtk
            velfieldToUse = self.velfield
            pointLocator_fluidmeshToUse = self.pointLocator_fluidmesh
            applyRotationAfterwards = True
        
        # Construct the velocity field to use as pandas dataframe
        if hasattr(self, 'velfieldPD') and self.useSphere:
            velfieldToUsePD = pd.DataFrame(velfieldToUse,
                                           index=None,
                                           columns=self.velfieldPD.columns
                                           )

# =============================================================================
#         # 3. Get the fluid cell points which are used
# =============================================================================
        if self.useSphere:
            # In the case of the extrapolation, we use a cylinder of points
            # since it is not possible to determine the number of steps
            # outward to take before interpolation is succesful.
            if self.velMethod == '0':
                maxH = stepSizeAlongNormal  * maxNStepsAlongNormal * 1.5
                self.maxH = maxH
                np_fluidmeshpoints_ids = self._find_ids_in_column_with_Cylinder(fluidmeshToUse,
                                                                                self.R,
                                                                                np.array([0., 0., 0.]),
                                                                                maxH,
                                                                                upVector)
            else:
                # When the boundary layer edge must be determined we use
                # a larger column of points to operate on
                if findBL:
                      np_fluidmeshpoints_ids = self._find_ids_in_sphere_with_Locator(pointLocator_fluidmeshToUse,
                                                                                     self.R,
                                                                                     np.array([0., 0., 0.])
                                                                                     )
                else:
                    np_fluidmeshpoints_ids = self._find_ids_in_sphere_with_Locator(pointLocator_fluidmeshToUse,
                                                                                   self.R,
                                                                                   vtk_point + shiftVectorLocal
                                                                                   )
        else:
            NCoins = 2
            cylinderHeight = self.coinHeight + self.coinHeight * (NCoins-1) * (1 - self.coinOverlap/100)
            self.cylinderHeight = cylinderHeight
            dzCoin = (cylinderHeight - self.coinHeight) / (NCoins - 1)
            cylinderbottom = vtk_point + shiftVectorLocal - self.coinHeight/2 * cellNormal
            np_fluidmeshpoints_ids = self._find_ids_in_column_with_Cylinder(fluidmeshToUse,
                                                                            self.R,
                                                                            cylinderbottom,
                                                                            cylinderHeight,
                                                                            cellNormal,
                                                                            rotMat)

        if (self.velMethod == '1') and findBL:
            maxH = stepSizeAlongNormal  * maxNStepsAlongNormal * 1.5
            np_fluidmeshpointsColumn_ids = self._find_ids_in_column_with_Cylinder(fluidmeshToUse,
                                                                                  self.R,
                                                                                  np.array([0., 0., 0.]),
                                                                                  maxH,
                                                                                  upVector)
            
        # Apply slicer
        # Update velfield in sphere
        velcoor_to_slice = velfieldToUse[np_fluidmeshpoints_ids, :]
        self.velcoor_to_slice_pre = velcoor_to_slice 
        
        if applyRotationAfterwards:
            # 1. Apply shift
            velcoor_to_slice[:,:3] -= shiftVectorLocal
            # 2. Mask all points below 0.
            zmask = velcoor_to_slice[:,2] >= 0.
            velcoor_to_slice = velcoor_to_slice[zmask , :]
            np_fluidmeshpoints_ids = np.array(np_fluidmeshpoints_ids)[zmask]
            # 3. Rotate the true velfield as needed
            velcoor_to_slice[:,:3] = funcRotateForward(velcoor_to_slice[:,:3] - vtk_point)
            self.velcoor_to_slice_post = velcoor_to_slice
        
        velcoorMask = self.sliceFunc(*velcoor_to_slice[:,:3].T)
        np_fluidmeshpoints_ids = np_fluidmeshpoints_ids[velcoorMask]
        velfield_in_sphere = velcoor_to_slice[velcoorMask, :]

        NFMPoints = len(np_fluidmeshpoints_ids)
        
        
        # self.zmask = zmask
        # self.np_fluidmeshpoints_ids = np_fluidmeshpoints_ids
        self.NFMPoints = NFMPoints
        self.np_fluidmeshpoints_ids = np_fluidmeshpoints_ids
        # self.velfieldToUse
        
        if self.useCoin and not self.coinFittingMethod=='LIN':
            # Divide this up in to the given number of coins
            FluidMeshPointIdsInCoin = []
            for coini in range(NCoins):
                velfieldValuesToUse = velfieldToUse[np_fluidmeshpoints_ids, 2]
                lowBound = coini * dzCoin - self.coinHeight/2
                highBound = coini * dzCoin + self.coinHeight/2
                print(f'{lowBound = } | {highBound =}')
                FluidMeshPointIdsInCoin.append(np_fluidmeshpoints_ids[(velfieldValuesToUse >= lowBound) &
                                                                      (velfieldValuesToUse < highBound)
                                                                      ]
                                                )
# =============================================================================
#         # 4. Get the object mesh points which are used        
# =============================================================================
        if self.useCoin:
            self.objectIntersect = None
            if self.useObjectInfo:
                objmeshpoints = np.array([0., 0., 0.])
                NOMPoints = 1
            else:
                NOMPoints = 0
        else:
            # If needed, find number of object mesh points and its distribution
            if self.useObjectInfo and (not self.LSR_constrained) and self.useSphere:
                # Find the section of the circle which
                # 1. Intersects with the ground
                ground_intersect = self._find_ground_intersect(vtk_point,
                                                               self.R,
                                                               z_ground=0.)
                
                # 2. Intersects with the model
                # Check if sphere intersects with object itself
                closest_point = np.array([0.0, 0.0, 0.0])
                projected_cellId_obj = vtk.reference(0)
                subId_obj = vtk.reference(0)
                dist2_obj = vtk.reference(0.)
                cell_obj = vtk.vtkGenericCell()
                model_intersect = vtk.reference(0)
                
                self.cellLocator_objectSTL.FindClosestPointWithinRadius(vtk_point,
                                                                        self.R,
                                                                        closest_point,
                                                                        cell_obj,
                                                                        projected_cellId_obj,
                                                                        subId_obj,
                                                                        dist2_obj,
                                                                        model_intersect
                                                                        )
                
                # Check if closer to ground or model
                closerToGround = (not model_intersect) or (vtk_point[2] <= np.sqrt(dist2_obj.get()))
                
                if (ground_intersect is not None) and closerToGround:
                    Vs = self._VolumeCutSphere(self.R, self.R - vtk_point[2])                    
                    # Relation 4 for determining Num OM-points
                    # For griddata_mode, we simply use 3 points
                    if self.griddata_mode:
                        NOMPoints = 3
                    else:
                        NOMPoints = self._NOMPoints_from_NFMPoints(NFMPoints, ground_intersect[1])
                        # round(NFMPoints * ground_intersect[1] / (Vs *3 / (4*np.pi))**(1/3))
                    
                    # Using a uniform distribution, place N points inside the circle of intersection
                    groundCenter = np.array([vtk_point[0], vtk_point[1], vtk_point[2]])
                    groundCircleRadius = self.R
                    
                    # Make a switch.
                    # a. When only the ground plane is cut, then use a sunflower distribution
                    # if not model_intersect:
                    objmeshpoints = self._sunflower_distribution(groundCenter,
                                                                 groundCircleRadius,
                                                                 NOMPoints)
                    # b. Else use a poisson Sampling distribution
                    # I probably chose this switch to later implement a merger of the model and
                    # ground plane which then requires a poisson sampling.
                    # For now I turn that off
                    # else:
                    #     objmeshpoints = self._poissonSampleDistributionCircle(groundCenter,
                    #                                                           groundCircleRadius,
                    #                                                           NOMPoints)
                    self.objmeshpoints = objmeshpoints
                    if self.griddata_mode:
                        self.objectIntersect = CreateVTKPolyDataFromTrianglePointsNP(objmeshpoints)
                    else:
                        self.objectIntersect = CreateVTKPolyDataAsCircle(groundCenter, groundCircleRadius)
                elif model_intersect:
                    sphereCenter = np.array(vtk_point)
                    
                    # When in griddata mode, only three points of the the cell are
                    # needed to define the objectmeshpoints
                    if self.griddata_mode:
                        objmeshpoints = numpy_support.vtk_to_numpy(vtk_cell.GetPoints().GetData())
                        NOMPoints = 3
                        
                        self.objectIntersect = CreateVTKPolyDataFromTrianglePointsNP(objmeshpoints)
                    # In quadratic regression mode (so not griddata), many more points are needed
                    else:
                        objmeshpoints, NOMPoints, self.objectIntersect = SamplePointsOnIntersection(self.modelPD_vtkUpSampled,
                                                                                              sphereCenter,
                                                                                              self.R,
                                                                                              NFMPoints,
                                                                                              self.griddata_mode,
                                                                                              self.C,
                                                                                              returnIntersect=True
                                                                                              )
                    # except 
                    self.objmeshpoints = objmeshpoints
                    
                else:
                    NOMPoints = 0
                    self.objectIntersect = None
            elif self.useObjectInfo:
                # Determine the closest point to add as constraint
                NOMPoints = 1
                self.objectIntersect = None
            else:
                NOMPoints = 0
                self.objectIntersect = None
                
        self.NOMPoints = NOMPoints
                
# =============================================================================
#         # 5. Get the velocity field inside the interrogation element
# =============================================================================
    
        if not self.useObjectInfo:
            # if self.velMethod == '0' or findBL:
                # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
            # else:
                # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                # # 1. Apply shift
                # velfield_in_sphere[:,:3] -= shiftVectorLocal
                # # 2. Mask all points below 0.
                # velfield_in_sphere = velfield_in_sphere[velfield_in_sphere[:,2] >= 0., :]
                # # 3. Rotate the true velfield as needed
                # velfield_in_sphere[:,:3] = funcRotateForward(velfield_in_sphere[:,:3] - vtk_point)
            
            if hasattr(self, 'vorticityNameX') and (self.velMethod=='0' or findBL):
                velfield_in_spherePD = velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :]
            
            # velfield_in_sphere[:,:3] -= vtk_point
            self.velfieldColumn = velfield_in_sphere
            # Construct the interpolator beforehand, so we only have to do it once
            if test:
                tCreateInterp0 = time.time()
            
            try:
                gridDataInterp = self._estimate_velocity_at_sphere_center_with_griddata(velfield_in_sphere[:,:3],
                                                                                        velfield_in_sphere[:,3:6],
                                                                                        special=special, returnInterp=True)
            except sp.spatial.qhull.QhullError as err:
                if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                    warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                # Skip this point
                

            if test:
                print(f'\tIt took {time.time()-tCreateInterp0:.3f} seconds to compute interpolator')
        elif self.useSphere:
            if self.LSR_constrained:
                if findBL:
                    # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                    if hasattr(self, 'vorticityNameX'):
                        velfield_in_spherePD = velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :]
                else:
                    # # 0. Use all velfield nodes inside sphere
                    # velfield_in_sphere = velfieldToUse[np_fluidmeshpoints_ids, :]
                    # # 1. Apply shift
                    # velfield_in_sphere[:,:3] -= shiftVectorLocal
                    # # 2. Mask all points below 0.
                    # velfield_in_sphere = velfield_in_sphere[velfield_in_sphere[:,2] >= 0., :]
                    # # 3. Rotate the true velfield as needed
                    # velfield_in_sphere[:,:3] = funcRotateForward(velfield_in_sphere[:,:3] - vtk_point)
                    if hasattr(self, 'vorticityNameX'):
                        # 0. Use all velfield nodes inside sphere
                        velfield_in_spherePD = velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :]
                        # 1. Apply shift
                        velfield_in_spherePD.iloc[:,:3] -= shiftVectorLocal
                        # 2. Mask all  points below 0.
                        velfield_in_spherePD.iloc[:,:3] = velfield_in_spherePD.loc[velfield_in_spherePD.iloc[:,2] >= 0., :]
                        # 3. Rotate the true velfield as needed
                        velfield_in_spherePD.iloc[:,:3] = funcRotateForward(velfield_in_spherePD.iloc[:,:3] - vtk_point)
                
            else:
                # Swap the vstack with a pre-memory allocation
                velfield_in_sphereUse = np.empty((NFMPoints + NOMPoints, self.NExtraVelfieldValues + 3))
                # Populate with fluid mesh points
                # if findBL:
                #     velfield_in_sphere[:NFMPoints, :] = velfieldToUse[np_fluidmeshpoints_ids, :]
                # else:
                    
                    # 0. Use all velfield nodes inside sphere
                    # velfield_in_sphereNFM = velfieldToUse[np_fluidmeshpoints_ids, :]
                    # # # 1. Apply shift
                    # velfield_in_sphereNFM[:,:3] -= shiftVectorLocal
                    # # 2. Mask all points below 0.
                    # velfield_in_sphereNFM = velfield_in_sphereNFM[velfield_in_sphereNFM[:,2] >= 0., :]
                    # # 3. Rotate the true velfield as needed
                    # velfield_in_sphereNFM[:,:3] = funcRotateForward(velfield_in_sphereNFM[:,:3] - vtk_point)
                    
                velfield_in_sphereUse[:NFMPoints, :] = velfield_in_sphere
                
                # Populate with object mesh points
                velfield_in_sphereUse[NFMPoints:, :3] = funcRotateForward(objmeshpoints - vtk_point)
                velfield_in_sphereUse[NFMPoints:, 3:] = 0.
                
                # Overwrite
                velfield_in_sphere = velfield_in_sphereUse
                
                if self.velMethod == '1' and hasattr(self, 'vorticityNameX') and findBL:
                    velfield_in_column = velfieldToUse[np_fluidmeshpointsColumn_ids, :]
                    
                    velfield_in_columnPD = pd.DataFrame(velfield_in_column,
                                                        index=None,
                                                        columns=self.velfieldPD.columns
                                                        )
                
                if hasattr(self, 'vorticityNameX'):
                    zerosDF = pd.DataFrame(np.c_[funcRotateForward(objmeshpoints - vtk_point - shiftVectorLocal),
                                                 np.zeros((NOMPoints, self.NExtraVelfieldValues))],
                                           index=None,
                                           columns=self.velfieldPD.columns
                                           )
                    if findBL:
                        velfield_in_spherePD = pd.concat([velfieldToUsePD.iloc[np_fluidmeshpoints_ids, :], zerosDF],
                                                             ignore_index=True)
                    else:
                        velfieldToUsePDlocal = pd.DataFrame(velfield_in_sphere,
                                                            index=None,
                                                            columns=self.velfieldPD.columns
                                                           )
                        velfield_in_spherePD = pd.concat([velfieldToUsePDlocal, zerosDF],
                                                         ignore_index=True)
        
        self.velfield_in_sphere = velfield_in_sphere
        self.velfield_in_sphereTrue = funcRotateBackward(velfield_in_sphere[:, :3]) + vtk_point
        
# =============================================================================
#         # 6. Approximate the velocity        
# =============================================================================
        if self.useCoin:
            # Use the coin to compute the velocity in each coin with a linearised mean            
        
            if not self.coinFittingMethod=='LIN':
                velocityAlongNormal = np.zeros((NCoins,3))
                for coini in range(NCoins):
                    velfieldInCoin = velfieldToUse[FluidMeshPointIdsInCoin[coini], :]
                    # velfieldValuesInCoin = funcRotateForward(velfieldInCoin[:,3:6])
                    velfieldValuesInCoin = velfieldInCoin[:,3:6]
                    NpInCoin = np.ma.size(velfieldInCoin, axis=0)
                    centerCoin = dzCoin * coini
                    
                    # Estimate mean u-velocity
                    velocityAlongNormal[coini, 0], _ = self._estimateLinearMeanInCoin(centerCoin,
                                                                                      velfieldInCoin[:,:3],
                                                                                      velfieldValuesInCoin[:,0],
                                                                                      NpInCoin)
                    # Estimate mean v-velocity
                    velocityAlongNormal[coini, 1], _ = self._estimateLinearMeanInCoin(centerCoin,
                                                                                                   velfieldInCoin[:,:3],
                                                                                                   velfieldValuesInCoin[:,1],
                                                                                                   NpInCoin)
                    # Estimate mean w-velocity
                    velocityAlongNormal[coini, 2], _ = self._estimateLinearMeanInCoin(centerCoin,
                                                                                                              velfieldInCoin[:,:3],
                                                                                                              velfieldValuesInCoin[:,2],
                                                                                                              NpInCoin)
            else:
                velocityAlongNormal = np.zeros((1,3))
                dVel_dStep = np.zeros((3))
                velfieldInCoin = velfield_in_sphere#velfieldToUse[np_fluidmeshpoints_ids, :]
                
                # Estimate mean u-velocity
                velocityAlongNormal[0, 0], _, dVel_dStep[0] = self._estimateLinearMeanInCoin(None,
                                                                                             velfieldInCoin[:,:3],
                                                                                             velfieldInCoin[:,3],
                                                                                             NFMPoints,
                                                                                             returnGradient=True)
                
                # Estimate mean v-velocity
                velocityAlongNormal[0, 1], _, dVel_dStep[1] = self._estimateLinearMeanInCoin(None,
                                                                                             velfieldInCoin[:,:3],
                                                                                             velfieldInCoin[:,4],
                                                                                             NFMPoints,
                                                                                             returnGradient=True)
                
                # Estimate mean w-velocity
                velocityAlongNormal[0, 2], _, dVel_dStep[2] = self._estimateLinearMeanInCoin(None,
                                                                                             velfieldInCoin[:,:3],
                                                                                             velfieldInCoin[:,5],
                                                                                             NFMPoints,
                                                                                             returnGradient=True)
            # nrange = np.array([0, dzCoin])
            self.velocityAlongNormal = velocityAlongNormal
            
            if self.coinFittingMethod=='LIN':
                self.vel = velocityAlongNormal[0, :]
                self.u, self.v, self.w = self.vel
                
                self.dzCoin = dzCoin
                velWallNormalGradient = dVel_dStep * 1e3
                self.normalGradient = velWallNormalGradient 
                self.velWallNormalGradientInPlane = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                (cellNormal).reshape((1,3)))
            
            # Fit a user-specified function to the mean velocities
            # Option: No-fitting, just take it directly from the linearised mean
            elif self.coinFittingMethod == 'NONE':
                self.vel = velocityAlongNormal[int(self.useObjectInfo), :]
                
            
            # Option: Fit a linear function through it all
            else:
                vel = np.zeros((3,))
                dVel_dStep = np.zeros((3,))
                centerCoin = cylinderHeight / 2
                velfieldInCylinder = velfieldToUse[np_fluidmeshpoints_ids, :]
                # Estimate mean u-velocity
                vel[0], dVel_dStep[0], _ = self._estimateLinearMeanInCylinder(np.array([0., 0., 0.,]),
                                                                              velfieldInCylinder[:,:3],
                                                                              velfieldInCylinder[:,3],
                                                                              NFMPoints,
                                                                              projectedPointTranslated = np.array([0., 0., 0.])
                                                                              )
                
                # Estimate mean v-velocity
                vel[1], dVel_dStep[1], _ = self._estimateLinearMeanInCylinder(np.array([0., 0., 0.,]),
                                                                              velfieldInCylinder[:,:3],
                                                                              velfieldInCylinder[:,4],
                                                                              NFMPoints,
                                                                              projectedPointTranslated = np.array([0., 0., 0.])
                                                                              )
                
                # Estimate mean w-velocity
                vel[2], dVel_dStep[2], _ = self._estimateLinearMeanInCylinder(np.array([0., 0., 0.,]),
                                                                              velfieldInCylinder [:,:3],
                                                                              velfieldInCylinder[:,5],
                                                                              NFMPoints,
                                                                              projectedPointTranslated = np.array([0., 0., 0.])
                                                                              )
                self.vel = vel
                self.u[c_idx], self.v[c_idx], self.w[c_idx] = vel
                velWallNormalGradient = dVel_dStep * 1e3
                self.velWallNormalGradientInPlane[c_idx, :] = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                          (cellNormal).reshape((1,3)))
            
            
        elif self.griddata_mode:
            if ((len(velfield_in_sphere[:,0] ) >= 4) and (not self.volume_mode)):
                if not self.useObjectInfo:
                    # Can very likely not compute velocity, but we can try and extrapolate velocity to approximate this at the cell location
                    # then we can compute also tau, shape-factor and the rest
                    # We need to slowly move along the cell normal to find the first valid point to then extrapolate this towards the outputmesh
                    pointIsInvalid = True
                    pointAlongNormal = np.array([0., 0., 0.])
                    stepCount = 0
                    while pointIsInvalid and (stepCount < maxNStepsAlongNormal):
                        # Shift the velocity field (this is as if we are moving along the normal)
                            
                        # Attempt to compute velocity
                        try:
                            # Execute function
                            velRotated = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                velfield_in_sphere[:,3:6],
                                                                                                griddata_interp=gridDataInterp,
                                                                                                evaluationPoint=np.c_[pointAlongNormal].T,
                                                                                                special=special)
                            
                        except sp.spatial.qhull.QhullError as err:
                            if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                            # Mark the velocity invalid
                            velRotated = (np.nan, np.nan, np.nan)
                                            

                            
                        if ~np.isnan(velRotated[0]):
                            pointIsInvalid = False
                            self.isValid = True
                            
                            ###########
                            ## If velocity is valid, we compute omega
                            if hasattr(self, 'vorticityNameX'):
                                try:
                                    omega = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                   velfield_in_spherePD.loc[:,self.omegaColumns].to_numpy(),
                                                                                                   griddata_interp=gridDataInterp,
                                                                                                   evaluationPoint=np.c_[pointAlongNormal].T,
                                                                                                   special=special)

                                except sp.spatial.qhull.QhullError as err:
                                    if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                        warnings.warn(f'Qhull error occurred for {c_idx = } with error: "{err}"')
                                        break
                                    omega = (np.nan, np.nan, np.nan)
                                                        
                            # Get global point
                            globalPointAlongNormal = (funcRotateBackward(pointAlongNormal) + vtk_point).flatten()
                            if test:
                                print(f'\tFirst valid velocity-point found at {globalPointAlongNormal}.')
                        else:
                            ### If function fails, then velocity is invalid and we try again
                            # Update the pointAlongNormal
                            pointAlongNormal += stepSizeAlongNormal * upVector

                            # Update the step count
                            stepCount += 1
                    
                    ### If point is valid, we compute the dynamic parameters
                    # Compute the velocity one step further away
                    pointAlongNormalNext = pointAlongNormal + stepSizeAlongNormal * upVector * 1
                        
                    velRotatedNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                            velfield_in_sphere[:,3:6],
                                                                                            griddata_interp=gridDataInterp,
                                                                                            evaluationPoint=np.c_[pointAlongNormalNext].T,
                                                                                            special=special)
                    if hasattr(self, 'vorticityNameX'):
                        omegaNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                            velfield_in_spherePD.loc[:,self.omegaColumns].to_numpy(),
                                                                                            griddata_interp=gridDataInterp,
                                                                                            evaluationPoint=np.c_[pointAlongNormal].T,
                                                                                            special=special)

                    #a1. Compute velocity by extrapolation
                    dvelRotated = np.array(velRotatedNext) - np.array(velRotated)
                    dVel_dStep = dvelRotated / stepSizeAlongNormal
                    deltaStepAlongNormal = stepCount * stepSizeAlongNormal
                    velocityAtVTKPoint = np.array(velRotated) - deltaStepAlongNormal * dVel_dStep
                    vel = velocityAtVTKPoint
                    
                    #a2. Compute vorticity by extrapolation
                    if hasattr(self, 'vorticityNameX'):
                        dOmega = np.array(omegaNext) - np.array(omega)
                        dOmega_dStep = dOmega / stepSizeAlongNormal
                        omegaAtVTKPoint = np.array(omega) - deltaStepAlongNormal * dOmega_dStep
                    
                    #b. Compute the in-plane velocity wall-normal gradient components
                    velWallNormalGradient = dVel_dStep * 1e3
                    self.velWallNormalGradientInPlane = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                              (cellNormal).reshape((1,3)))
                    
                    
                    #c. Compute the delta* and theta parameters
                    #i. Compute the normal velocity profile until the approximated boundary layer height
                    # Define flag for boundary layer height, based on vorticity
                    if hasattr(self, 'vorticityNameX') and findBL:
                        vorticityNotConverged = True
                        stepCountVorticity = stepCount
                        initialStepFactor = 10
                        useLargeStep = True
                        pointAlongNormal = np.array([0., 0., 1.e-15])
                        if test:
                            print('Iterating to find boundary layer height')
                        while vorticityNotConverged and (stepCountVorticity < maxNStepsAlongNormal):
                            #### Condition 1: Based on a threshold value of the total vorticity
                            pointAlongNormalProfile = pointAlongNormal + stepSizeAlongNormal * upVector * (stepCountVorticity+1)
                            
                            # # Compute vorticity at the point using vtk's gradient filter
                            omegaNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                               velfield_in_spherePD.loc[:,self.omegaColumns].to_numpy(),
                                                                                               griddata_interp=gridDataInterp,
                                                                                               evaluationPoint=np.c_[pointAlongNormalProfile].T,
                                                                                               special=special)
                            omegaMag = np.linalg.norm(omegaNext)
                            
                            if test:
                                print(f'Computed vorticity at point ({np.round(pointAlongNormalProfile, 2)}): omegaMag = {np.round(omegaMag, 2)}')
                            
                            if omegaMag < omegaThreshold:
                                if useLargeStep and (stepCountVorticity >= initialStepFactor):
                                    useLargeStep = False
                                    stepCountVorticity += (1 - initialStepFactor)
                                    continue
                                # Converged
                                vorticityNotConverged = False
                                # Save the results
                                delta = (stepCountVorticity + 1) * stepSizeAlongNormal
                                self.delta = delta
                                self.BLvalid = True
                                if test:
                                    print(f'\tHeight of boundary layer estimated to be {delta:.1f}mm')
                                # Compute delta* and theta using the integral
                                
                                # and lastly the shape Factor
                                
                            else:
                                # stepCount based on large step or small step
                                if useLargeStep:
                                    stepCountVorticity += initialStepFactor
                                else:
                                    stepCountVorticity += 1
                        if self.delta == 0:
                            self.BLvalid = False
                            if test:
                                print('\tBoundary layer iterate not converged.')

                else:
                    # Compute the cell-centered velocity using griddata
                    
                    # 0. Build the interpolators
                    try:
                        gridDataInterp = self._estimate_velocity_at_sphere_center_with_griddata(velfield_in_sphere[:,:3],
                                                                                                velfield_in_sphere[:,3:6],
                                                                                                special=special, returnInterp=True)
                        if findBL:
                            gridDataInterpColumn = self._estimate_velocity_at_sphere_center_with_griddata(velfield_in_column[:,:3],
                                                                                                          velfield_in_column[:,3:6],
                                                                                                          special=special, returnInterp=True)
                    except sp.spatial.qhull.QhullError as err:
                        if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                            warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                    
                    #1. Velocity at point
                    try:
                        vel = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                     velfield_in_sphere[:,3:6],
                                                                                     griddata_interp=gridDataInterp,
                                                                                     evaluationPoint=np.array([[0, 0, 1e-15]]),
                                                                                     special=special)
                        
                        omegaAtVTKPoint = np.zeros((3,))
                        
                    except sp.spatial.qhull.QhullError as err:
                        if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                            warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                        vel = (np.nan, np.nan, np.nan)
                        
                    ###2. Velocity gradient
                    #a. We compute the velocity at point one step away from the current point
                    pointNext = stepSizeAlongNormal * upVector
                    try:
                        velNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                     velfield_in_sphere[:,3:6],
                                                                                     griddata_interp=gridDataInterp,
                                                                                     evaluationPoint=np.c_[pointNext].T,
                                                                                     special=special)
                    except sp.spatial.qhull.QhullError as err:
                        if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                            warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                    
                    #b. Compute the gradient using a forward difference scheme
                    dvel = np.array(velNext) - np.array(vel)
                    dVel_dStep = dvel / stepSizeAlongNormal
                    self.velWallNormalGradientInPlane = vel_tangent((dVel_dStep).reshape((1,3)) * 1e3,
                                                                              (cellNormal).reshape((1,3)))
                    
                    #3. Boundary layer height
                    #i. Compute the normal velocity profile until the approximated boundary layer height
                    # Define flag for boundary layer height, based on vorticity
                    if hasattr(self, 'vorticityNameX') and findBL:
                        vorticityNotConverged = True
                        stepCountVorticity = 0
                        initialStepFactor = 10
                        useLargeStep = True
                        pointAlongNormal = np.array([0., 0., 1.e-15])
                        # maxNStepsAlongNormal = self.R / stepSizeAlongNormal
                        if test:
                            print('Iterating to find boundary layer height')
                        while vorticityNotConverged and (stepCountVorticity < maxNStepsAlongNormal):
                            #### Condition 1: Based on a threshold value of the total vorticity
                            pointAlongNormalProfile = pointAlongNormal + stepSizeAlongNormal * upVector * (stepCountVorticity+1)

                            # # Compute vorticity at the point using griddata
                            try:
                                omegaNext = self._estimate_velocity_at_sphere_center_with_griddata(None,
                                                                                                   velfield_in_columnPD.loc[:,self.omegaColumns].to_numpy(),
                                                                                                   griddata_interp=gridDataInterpColumn,
                                                                                                   evaluationPoint=np.c_[pointAlongNormalProfile].T,
                                                                                                   special=special)
                            except sp.spatial.qhull.QhullError as err:
                                if (not ('QH6154 Qhull precision error' in err.args[0])) and (not ('QH6013 qhull input error' in err.args[0])):
                                    warnings.warn(f'Unexpected Qhull error occurred for {c_idx = } with error: "{err}"')
                                    
                                omegaNext = (np.nan, np.nan, np.nan)
                            omegaMag = np.linalg.norm(omegaNext)
                            
                            if test:
                                print(f'Computed vorticity at point ({np.round(pointAlongNormalProfile, 2)}): omegaMag = {np.round(omegaMag, 2)}')
                            
                            if omegaMag < omegaThreshold:
                                if useLargeStep and (stepCountVorticity >= initialStepFactor):
                                    useLargeStep = False
                                    stepCountVorticity += (1 - initialStepFactor)
                                    continue
                                # Converged
                                vorticityNotConverged = False
                                # Save the results
                                delta = (stepCountVorticity + 1) * stepSizeAlongNormal
                                self.BLvalid = True
                                self.delta = delta
                                if test:
                                    print(f'\tHeight of boundary layer estimated to be {delta:.1f}mm')
                                # Compute delta* and theta using the integral
                                
                                # and lastly the shape Factor
                                
                            else:
                                # stepCount based on large step or small step
                                if useLargeStep:
                                    stepCountVorticity += initialStepFactor
                                else:
                                    stepCountVorticity += 1
                        if self.delta == 0:
                            self.BLvalid = False
                            if test:
                                print('\tBoundary layer iterate not converged.')
                    
                
                # Assign the velocity
                self.u, self.v, self.w = vel
                
                # Assign the vorticity
                if hasattr(self, 'vorticityNameX'):
                    self.omegaX, self.omegaY, self.omegaZ = omegaAtVTKPoint

                # Determine if the velocity is valid
                self.isValid= ~np.isnan(vel[0])
        else:
            # Do not compute LSR if number of particles is less than 10.
            # This is not enough for LSR with 3*10 coefficients.
            
            if div_free:
                try:
                    vel, res = self._estimate_velocity_at_sphere_center_with_LSR_divfree(velfield_in_sphere[:,:3],
                                                                                         velfield_in_sphere[:,3:6],
                                                                                         order=self.LSR_ORDER)
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        pass
                    else:
                        raise NotImplementedError(f'Different error occurred:\n{str(err)}')

                self.u, self.v, self.w = vel
                res_u, res_v, res_w = res
            else:
                
                ####### Here we execute method 4 #######
                if self.LSR_constrained:
                    # Check if sphere intersects ground
                    dist2ground = vtk_point[2]
                    if dist2ground <= self.R:
                    # if (vtk_point[2] - self.R) <= 0.: #sphere intersects ground
                        sphere_intersects_ground = True
                    else:
                        sphere_intersects_ground = False
                        
                    # Check if sphere intersects with object itself
                    closest_point = np.array([0.0, 0.0, 0.0])
                    projected_cellId_obj = vtk.reference(0)
                    subId_obj = vtk.reference(0)
                    dist2_obj = vtk.reference(0.)
                    cell_obj = vtk.vtkGenericCell()
                    sphere_intersects_model = vtk.reference(0)
                    
                    self.cellLocator_objectSTL.FindClosestPointWithinRadius(vtk_point,
                                                                            self.R,
                                                                            closest_point,
                                                                            cell_obj,
                                                                            projected_cellId_obj,
                                                                            subId_obj,
                                                                            dist2_obj,
                                                                            sphere_intersects_model)
                    dist_obj = np.sqrt(dist2_obj.get())
                    
                    # Combine checks to see if wall constraint needs to be applied
                    sphere_intersects_surface = sphere_intersects_ground or sphere_intersects_model
                    Use_sphere_intersects_ground = (dist2ground < dist_obj) and sphere_intersects_ground
                    Use_sphere_intersects_model = (dist2ground >= dist_obj) and sphere_intersects_model
                else:
                    sphere_intersects_ground = False
                    Use_sphere_intersects_ground = False
                    Use_sphere_intersects_model = False
                    
                    
                if (self.LSR_constrained) and sphere_intersects_surface:
                    ## Estimate velocity using constrained optimization
                    # 1a. Get the projected point (check for ground intersection)
                    if Use_sphere_intersects_ground:
                        projected_point = np.array([vtk_point[0], vtk_point[1], 0]) - vtk_point
                        # self.projected_point_ground = projected_point_ground
                        
                    # 1b. Get the projected point (check for model intersection)
                    if Use_sphere_intersects_model:
                        # 1. Get the projected point and cell in the objectmesh
                        projected_point = closest_point.astype(np.float32) - vtk_point
                    
                    # Execute constrained Least-Squares
                    solution = pivOpt.optimizeLSR_wCons(velfield_in_sphere[:,:3],
                                                        velfield_in_sphere[:,3:6],
                                                        projected_point,
                                                        use_all_constraints=False
                                                        )

                    # Compute the velocity at the center point
                    beta = solution['x']
                    self.beta = beta
                    self.vel = beta[[0,10,20]].flatten()
                    self.u, self.v, self.w = beta[[0,10,20]].flatten()
                    
                    # Extract the derivatives with respect to x at the center point
                    self.dudx, self.dvdx, self.dwdx = funcRotateBackward(np.c_[beta[[1,11,21]]].T).flatten()
                    # Extract the derivatives with respect to x at the center point
                    self.dudy, self.dvdy, self.dwdy = funcRotateBackward(np.c_[beta[[2,12,22]]].T).flatten()
                    # Extract the derivatives with respect to x at the center point
                    self.dudz, self.dvdz, self.dwdz = funcRotateBackward(np.c_[beta[[3,13,23]]].T).flatten()
                    
                    # Compute the vorticity and store
                    omega_x = self.dwdy - self.dvdz
                    omega_y = self.dudz - self.dwdx
                    omega_z = self.dvdx - self.dudy
                    self.omega = np.array([omega_x, omega_y, omega_z])
                    
                    # Store the wall-normal gradients
                    dVel_dStep = np.array([beta[3], beta[13], beta[23]])
                    velWallNormalGradient = dVel_dStep * 1e3
                    self.normalGradient = velWallNormalGradient
                    self.velWallNormalGradientInPlane = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                              (cellNormal).reshape((1,3)))
                    
                    # # Compute residuals in u
                    # res_u = velfield_in_sphere[:, 3] - pivOpt.vel3D_func_fast(beta[:10], velfield_in_sphere[:, :3]).T
                    # # In v
                    # res_v = velfield_in_sphere[:, 4] - pivOpt.vel3D_func_fast(beta[10:20], velfield_in_sphere[:, :3]).T
                    # # And in w
                    # res_w = velfield_in_sphere[:, 5] - pivOpt.vel3D_func_fast(beta[20:30], velfield_in_sphere[:, :3]).T

                else:
                    ####### Here we execute methods 2 and 3 ####### (Omg so much bs before this.....)
                    # estimate velocity-u
                    (self.u, 
                     beta_u, res_u) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphere[:,:3],
                                                                                        velfield_in_sphere[:,3],
                                                                                        order=self.LSR_ORDER)
                    # estimate velocity-v
                    (self.v,
                     beta_v, res_v) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphere[:,:3],
                                                                                        velfield_in_sphere[:,4],
                                                                                        order=self.LSR_ORDER)
                    # estimate velocity-w
                    (self.w,
                     beta_w, res_w) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphere[:,:3],
                                                                                        velfield_in_sphere[:,5],
                                                                                        order=self.LSR_ORDER)
                    self.vel = [self.u, self.v, self.w]
                    self.beta = np.c_[beta_u, beta_v, beta_w].T                                         
                    # Get the gradient coefficients from beta
                    self.dudx, self.dudy, self.dudz = funcRotateBackward(np.c_[beta_u[1:4]].T).flatten() * 1e3
                    self.dvdx, self.dvdy, self.dvdz = funcRotateBackward(np.c_[beta_v[1:4]].T).flatten() * 1e3
                    self.dwdx, self.dwdy, self.dwdz = funcRotateBackward(np.c_[beta_w[1:4]].T).flatten() * 1e3
                    
                    # Compute the vorticity and store
                    omega_x = self.dwdy - self.dvdz
                    omega_y = self.dudz - self.dwdx
                    omega_z = self.dvdx - self.dudy
                    self.omega = np.array([omega_x, omega_y, omega_z])
                    
                    # Store coefficients
                    self.beta = np.r_[beta_u, beta_v, beta_w]
                    
                    # Store the wall-normal gradients
                    dVel_dStep = np.array([beta_u[3], beta_v[3], beta_w[3]])
                    velWallNormalGradient = dVel_dStep * 1e3
                    self.normalGradient = velWallNormalGradient
                    self.velWallNormalGradientInPlane = vel_tangent((velWallNormalGradient).reshape((1,3)),
                                                                              (cellNormal).reshape((1,3)))
                    
                    # Now walk along the normal and compute the vorticity to threshold
                    if hasattr(self, 'vorticityNameX') and findBL:
                        vorticityNotConverged = True
                        stepCountVorticity = 0
                        initialStepFactor = 10
                        useLargeStep = True
                        pointAlongNormal = np.array([0., 0., 0])
                        if test:
                            print('Iterating to find boundary layer height')
                        while vorticityNotConverged and (stepCountVorticity < maxNStepsAlongNormal):
                            #### Condition 1: Based on a threshold value of the total vorticity
                            pointAlongNormalProfile = pointAlongNormal + stepSizeAlongNormal * upVector * (stepCountVorticity+1)

                            ###############################################
                            ########### A. Determine the fluidmesh ids
                            ###############################################
                            # 1. Determine velfield ids
                            np_fluidmeshpoints_idsNext = self._find_ids_in_sphere_with_Locator(pointLocator_fluidmeshToUse,
                                                                                               self.R,
                                                                                               pointAlongNormalProfile)
                            NFMPointsNext = np.size(np_fluidmeshpoints_idsNext)
                            
                            ###############################################
                            ########### B. Determine the objectmesh points
                            ###############################################
                            # Find the section of the circle which intersects
                            # 1. Intersects with the ground
                            ground_intersect = self._find_ground_intersect(pointAlongNormalProfile,
                                                                           self.R,
                                                                           z_ground=0.)
                            
                            # 2. Intersects with the model
                            # Check if sphere intersects with object itself
                            closest_point = np.array([0.0, 0.0, 0.0])
                            projected_cellId_obj = vtk.reference(0)
                            subId_obj = vtk.reference(0)
                            dist2_obj = vtk.reference(0.)
                            cell_obj = vtk.vtkGenericCell()
                            model_intersect = vtk.reference(0)
                            
                            self.cellLocator_objectSTL.FindClosestPointWithinRadius(pointAlongNormalProfile,
                                                                                    self.R,
                                                                                    closest_point,
                                                                                    cell_obj,
                                                                                    projected_cellId_obj,
                                                                                    subId_obj,
                                                                                    dist2_obj,
                                                                                    model_intersect
                                                                                    )
                            
                            # Check if closer to ground or model
                            closerToGround = (not model_intersect) or (pointAlongNormalProfile[2] <= np.sqrt(dist2_obj.get()))
                            
                            if (ground_intersect is not None) and closerToGround:
                                Vs = self._VolumeCutSphere(self.R, self.R - pointAlongNormalProfile[2])  
                                
                                NOMPointsNext = self._NOMPoints_from_NFMPoints(NFMPointsNext, ground_intersect[1])
                                # round(NFMPointsNext  * ground_intersect[1] / (Vs *3 / (4*np.pi))**(1/3))
                                
                                # Using a poisson sample distribution,
                                # place N points inside the circle of intersection
                                groundCenter = np.array([pointAlongNormalProfile[0],
                                                         pointAlongNormalProfile[1],
                                                         0.])
                                groundCircleRadius = self.R
                                objmeshpointsNext = self._poissonSampleDistributionCircle(groundCenter,
                                                                                          groundCircleRadius,
                                                                                          NOMPointsNext)

                            elif model_intersect:
                                (objmeshpointsNext,
                                 NOMPointsNext) = SamplePointsOnIntersection(self.modelPD_vtkUpSampled,
                                                                             pointAlongNormalProfile,
                                                                             self.R,
                                                                             NFMPointsNext ,
                                                                             self.griddata_mode,
                                                                             self.C
                                                                             )

                            # Combine all together
                            velfield_in_sphereNext = np.vstack((velfieldToUse[np_fluidmeshpoints_idsNext, :], 
                                                                np.c_[funcRotateForward(objmeshpointsNext),
                                                                      np.zeros((NOMPointsNext, self.NExtraVelfieldValues))] ))
                            velfield_in_sphereNext[:,:3] -= pointAlongNormalProfile

                            # # Compute vorticity at the point using griddata
                            (_, beta_uNext, _) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphereNext[:,:3],
                                                                                                   velfield_in_sphereNext[:,3],
                                                                                                   order=self.LSR_ORDER)
                            # estimate velocity-v
                            (_, beta_vNext, _) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphereNext[:,:3],
                                                                                                   velfield_in_sphereNext[:,4],
                                                                                                   order=self.LSR_ORDER)
                            # estimate velocity-w
                            (_, beta_wNext, _) = self._estimate_velocity_at_sphere_center_with_LSR(velfield_in_sphereNext[:,:3],
                                                                                                   velfield_in_sphereNext[:,5],
                                                                                                   order=self.LSR_ORDER)
                            
                            # Get the gradient coefficients from beta
                            _,    dudy, dudz = funcRotateBackward(np.c_[beta_uNext[1:4]].T).flatten() * 1e3
                            dvdx, _,    dvdz = funcRotateBackward(np.c_[beta_vNext[1:4]].T).flatten() * 1e3
                            dwdx, dwdy, _    = funcRotateBackward(np.c_[beta_wNext[1:4]].T).flatten() * 1e3
                            
                            # Compute the vorticity and store
                            omega_x = dwdy - dvdz
                            omega_y = dudz - dwdx
                            omega_z = dvdx - dudy
                            omegaNext = np.array([omega_x, omega_y, omega_z])
                            
                            omegaMag = np.linalg.norm(omegaNext)
                            
                            if omegaMag < omegaThreshold:
                                if useLargeStep and (stepCountVorticity >= initialStepFactor):
                                    useLargeStep = False
                                    stepCountVorticity += (1 - initialStepFactor)
                                    continue
                                # Converged
                                vorticityNotConverged = False
                                # Save the results
                                delta = (stepCountVorticity + 1) * stepSizeAlongNormal
                                # Compute delta* and theta using the integral
                                
                                # and lastly the shape Factor
                                
                            else:
                                # stepCount based on large step or small step
                                if useLargeStep:
                                    stepCountVorticity += initialStepFactor
                                else:
                                    stepCountVorticity += 1
                                    
        # 8. Display the results in some way
        if plotResults:
            self._plot_after_evaluation()
                        
        pass
    
    def GetNearestCellInObjectSTL(self, point):
        if not hasattr(self, 'cellLocator_objectSTL'):
            self.cellLocator_objectSTL = self._build_vtkCellLocator(self.modelPD_vtk)
        
        closest_point = [0., 0., 0.]
        subId = vtk.reference(0)
        dist2 = vtk.reference(0.)
        out2 = vtk.reference(0)

        self.cellLocator_objectSTL.FindClosestPoint(point, closest_point, out2, subId, dist2)
        return out2, self.modelPD_vtk.GetCell(out2)
        
    def GetNearestCellInOutputMesh(self, point):
        if not hasattr(self, 'cellLocator_outputmesh'):
            self.cellLocator_outputmesh = self._build_vtkCellLocator(self.outputmesh_vtk)
        
        closest_point = [0., 0., 0.]
        subId = vtk.reference(0)
        dist2 = vtk.reference(0.)
        out2 = vtk.reference(0)

        self.cellLocator_outputmesh.FindClosestPoint(point, closest_point, out2, subId, dist2)
        return out2, self.outputmesh_vtk.GetCell(out2)
    
    def GetNearestPointInOutputMesh(self, point):
        p_idx = self.pointLocator_outputmesh.FindClosestPoint(point)
        point_coor = self.outputmesh_vtk.GetPoint(p_idx)
        
        return p_idx, point_coor
    
    def ShowSphereById(self, p_idx):
        assert hasattr(self, 'pointLocator_fluidmesh'), 'Can not show sphere, call BuildPointLocators() first.'
        
        # Get the point from the output mesh
        vtk_point = self.outputmesh_vtk.GetPoint(p_idx)
        
        # Find all fluid mesh point ids in the sphere
        FMpoint_ids = self._find_ids_in_sphere_with_Locator(self.pointLocator_fluidmesh, self.R, vtk_point)
        self.FMpoint_ids = FMpoint_ids
        
        # Find all object mesh point ids in the sphere
        if hasattr(self, 'objectmesh_vtk'):
            OMpoint_ids = self._find_ids_in_sphere_with_Locator(self.pointLocator_objectmesh, self.R, vtk_point)
            self.OMpoint_ids = OMpoint_ids
        
        # Find the corresponding fluid mesh coordinates
        try:
            FMcoor = self.velfield[FMpoint_ids, :3]
        except IndexError:
            warnings.warn('No fluid mesh nodes inside specified sphere.')
            FMcoor = None
        
        if hasattr(self, 'objectmesh_vtk') and (len(self.OMpoint_ids) > 0):
            # Find the corresponding object mesh coordinates
            OMcoor = self.objectmesh_np[OMpoint_ids, :]
            
            # Find all output mesh points in the sphere
            # Output the triangles which are inside the sphere
            OMtriangles_all = self._get_vtk_cells_by_pointIds(self.objectmesh_vtk,
                                                              OMpoint_ids)
            self.OMtriangles_all = OMtriangles_all
            
            # Create a set of the cell ids
            cell_lst = np.hstack(list(OMtriangles_all.values()))
            cell_set = set(cell_lst)
            
            # Loop over the set
            max_number_of_cells = len(cell_set)
            OMtriangles_valid = np.zeros((max_number_of_cells, 4))
            
            # Keep a count
            count = 0
            for cell_idx in cell_set:
                # Count if the cell occurs 3 times in the original list of triangles
                occurences = np.sum(cell_idx == cell_lst)
                if occurences == 3:
                    # Extract the cell from the outputmesh
                    vtk_cell = self.objectmesh_vtk.GetCell(cell_idx)
                    # Get point ids corresponding to the cell
                    vtk_cell_point_ids = vtk_cell.GetPointIds()
                    cell_point_ids = self._vtk_idlist_to_lst(vtk_cell_point_ids)
                    
                    # Add triangle to the list
                    OMtriangles_valid[count] = np.hstack((cell_idx, cell_point_ids))
                    count += 1
                else:
                    continue
            # Remove all unfilled entries
            if count != 0:
                OMtriangles_valid = np.array(OMtriangles_valid[:count, :].astype(np.int32).tolist())
                # OMtriangles_valid = list(zip(*self.OMtriangles_valid[:,1:].T.astype(np.int32)))
                self.OMtriangles_valid = OMtriangles_valid
            else:
                warnings.warn('No object mesh nodes inside specified sphere.')
                self.OMtriangles_valid = None
        
        # Define color in RGB(A)
        # Blue
        color1 = (0, 0, 0.8)
        color1_tp = (0, 0, 0.8, 0.5)
        
        # Orangey
        color2 = (1, 0.5, 0)
        color2_tp = (1, 0.5, 0, 0.5)
        
        # Green
        color3 = (0, 0.8, 0)
        color3_tp = (0, 0.8, 0, 0.5)
        
        # Create figure
        self.fig_sphere = plt.figure(figsize=(9,9))
        self.ax_sphere = self.fig_sphere.add_subplot(projection='3d')
        
        # Form the sphere
        t = np.linspace(0, 2*np.pi, 100)
        
        x1 = vtk_point[0] + self.R*np.cos(t)
        y1 = vtk_point[1] + self.R*np.sin(t)
        z1 = vtk_point[2] * np.ones_like(t)

        x2 = vtk_point[0] * np.ones_like(t)
        y2 = vtk_point[1] + self.R*np.sin(t)
        z2 = vtk_point[2] + self.R*np.cos(t)

        x3 = vtk_point[0] + self.R*np.sin(t)
        y3 = vtk_point[1] * np.ones_like(t)
        z3 = vtk_point[2] + self.R*np.cos(t)
        
        
        # Plot the data
        # Fluid mesh nodes
        if FMcoor is not None:
            self.ax_sphere.scatter(*FMcoor.T, color=color1, label='Fluid mesh', s=40)
        
        # Object mesh nodes
        if hasattr(self, 'objectmesh_vtk') and (len(self.OMpoint_ids) > 0):
            if not self.OMtriangles_valid is None:
                self.ax_sphere.plot_trisurf(*self.objectmesh_np.T,
                                            triangles=self.OMtriangles_valid[:,1:],
                                            color=color2_tp, ec=color2)
            self.ax_sphere.scatter(*OMcoor.T, color=color2, label='Object mesh', s=40)
        
        # Sphere
        self.ax_sphere.plot(*vtk_point, 'x-', markersize=10, color='red', label='Sphere')
        self.ax_sphere.plot(x1, y1, z1, color='red')
        self.ax_sphere.plot(x2, y2, z2, color='red')
        self.ax_sphere.plot(x3, y3, z3, color='red')
        
        # Format plot
        figure_title = f'Points inside sphere #{p_idx}'
        axis_title = f'Order of L-S: "x^{self.LSR_ORDER}" | Radius of spheres: {np.round(self.R, 3)}'
        xlabel = 'x [mm]'
        ylabel = 'y [mm]'
        zlabel = 'z [mm]'
        self.fig_sphere, self.ax_sphere = format_axis_style2(self.fig_sphere, self.ax_sphere,
                                                             xlabel, ylabel, zlabel,
                                                             suptitle=figure_title , axis_title=axis_title)
        # Define axis limits
        self.ax_sphere.set_xlim(x1.min(), x1.max())
        self.ax_sphere.set_ylim(y2.min(), y2.max())
        self.ax_sphere.set_zlim(z2.min(), z2.max())
        
        pass

def format_axis_style1(fig, axis, xlabel, ylabel,
                       suptitle=None, axis_title=None,
                       fs_large=24, fs_medium=20,
                       fs_small=16, labelsize=16,
                       add_legend=True):
    # Add suptitle
    if suptitle != None:
        fig.suptitle(suptitle, fontsize=fs_large)
        
    # Add axis title
    if axis_title != None:
        axis.set_title(axis_title, fontsize=fs_small)
        
    # Add the legend
    if add_legend:
        axis.legend(fontsize=fs_medium)
    
    # Add labels
    axis.set_xlabel(xlabel, fontsize=fs_medium)
    axis.set_ylabel(ylabel, fontsize=fs_medium)
    
    # Turn on minorticks
    axis.minorticks_on()
    # Resize ticks
    axis.tick_params(axis='both', which='major', labelsize=labelsize)
    # Add grid
    axis.grid(visible=True, which='major', color='black',
              linewidth=0.7)
    axis.grid(visible=True, which='minor', color='grey',
              linewidth=0.5, linestyle='-.')
    
    return fig, axis

def format_axis_style2(fig, axis, xlabel, ylabel, zlabel,
                       suptitle=None, axis_title=None,
                       fs_large=24, fs_medium=20,
                       fs_small=16, labelsize=16,
                       add_legend=True):
    # Add suptitle
    if suptitle != None:
        fig.suptitle(suptitle, fontsize=fs_large)
        
    # Add axis title
    if axis_title != None:
        axis.set_title(axis_title, fontsize=fs_small)
        
    # Add the legend
    if add_legend:
        axis.legend(fontsize=fs_medium)
    
    # Add labels
    axis.set_xlabel(xlabel, fontsize=fs_medium)
    axis.set_ylabel(ylabel, fontsize=fs_medium)
    axis.set_zlabel(zlabel, fontsize=fs_medium)
    
    return fig, axis

def compute_points_inside_stl_and_ground(points_to_check, stl_filepath, z_offset=0.0):
    # Find all points inside the stl
    _, _, stl_mask = enclosed_points_with_stl(points_to_check, stl_filepath)
    
    # Add points below ground (z_offset)
    mask = (points_to_check[:,2] < z_offset) | (stl_mask)
    points_outside = points_to_check[mask==0, :]
    points_inside = points_to_check[mask==1, :]
    
    return points_outside, points_inside, mask

def enclosed_points_with_stl(points_to_check, stl):
    if isinstance(stl, str):
        # Open the STL-file and get output (PolyData object)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl)
        reader.Update()
        stl_object = reader.GetOutput()
    else:
        stl_object = stl

    # Prepare the input numpy data for vtk
    assert np.ma.size(points_to_check, axis=1) == 3
    
    return maskCoordinatesInsideVTKPD(points_to_check, stl_object)

def time_function(f):
    @functools.wraps(f)
    def time_wrapper(*args, **kwargs):
        t0 = time.time()
        out = f(*args, **kwargs)
        t1 = time.time()
        print(f'func:{f.__name__} took: {np.round(t1-t0,1)} seconds.')
        return out
    return time_wrapper

class OverwriteException(Exception):
    pass

def plot_tetrahedrons(points, simplices, Nobj=3, fs=24, ls=16, z0=0, shift=np.zeros(3)):
    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection':'3d'})
    x = np.linspace(points[:,0].min()-2, points[:,0].max()+2, 10)
    y = np.linspace(points[:,1].min()-2, points[:,1].max()+2, 10)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx) * z0
    
    ax.plot_surface(xx, yy, zz, color='#FF00FF',  alpha=0.2)
    ax.plot([x.min(), x.min(), x.max(), x.max(), x.min()],
            [y.min(), y.max(), y.max(), y.min(), y.min()],
            [z0, z0, z0, z0, z0], color='#FF00FF')
    for tr in simplices:
        pts = points[tr, :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='c', lw='0.1')
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='c', lw='0.1')
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='c', lw='0.1')
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='c', lw='0.1')
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='c', lw='0.1')
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='c', lw='0.1')

    # Plot the fluid mesh
    ax.scatter(points[:-Nobj,0], points[:-Nobj,1], points[:-Nobj,2], color='b')
    
    # Plot the object mesh
    ax.scatter(points[-Nobj:,0], points[-Nobj:,1], points[-Nobj:,2], color='#FF00FF')
    ax.view_init(elev=14., azim=33.)
    ax.tick_params(axis='both', which='major', labelsize=ls)
    ax.xaxis.set_rotate_label(False)
    ax.set_xlabel(r'$X^{*}$ [mm]', fontsize=fs, rotation='horizontal')
    ax.xaxis.labelpad=20
    ax.set_ylabel(r'$Y^{*}$ [mm]', fontsize=fs)
    ax.yaxis.labelpad=15
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$Z^{*}$ [mm]', fontsize=fs, rotation='horizontal')
    ax.zaxis.labelpad=20
    ax.set_box_aspect(aspect=None, zoom=1.1)
    
    return fig, ax

class GridDataInterpolator(object):
    def __init__(self, p_in, timeit=False, show_done=False):
        # Save the input points
        self.p_in = p_in
        self.p_in_NoP = np.ma.size(p_in, axis=0)
        self.ndim = np.ma.size(p_in, axis=1)
        
        # Save the option to time all operations and show results
        self.timeit = timeit
        self.show_done = show_done
        
        # Initialise Delaunay triangulation object
        self.del_tri = None
        # Initialise object which stores simplex vertices
        self.vertices = None
        # Initiliase object which stores barycentric coordinates
        self.bary = None
        # Initiliase object which stores weights
        self.weights = None
    
    def Triangulate(self):
        '''
        Part one of a three part class computation.
        '''
        if self.timeit:
            t0 = time.time()
        self.del_tri = sp.spatial.Delaunay(self.p_in)
        if self.timeit:
            t1 = time.time()
        self.del_tri_NoP = np.ma.size(self.del_tri.simplices, axis=0)
        
        # Output results to user
        if self.show_done:
            # print(f'Delaunay triangulation completed. {self.del_tri_NoP:,} simplices formed using {self.p_in_NoP:,} points.')
            print(f'Delaunay triangulation completed. {self.del_tri_NoP:,} simplices formed using {self.p_in_NoP:,} points.')
            if self.timeit:
                print(f'Process took {np.round(t1-t0, 1)} seconds.')
                # print(f'Process took {np.round(t1-t0, 1)} seconds.')
        
        # End function
        pass
        
    def ComputeBarycentric(self, p_out, overwrite=False, batchmode=False, special=False, max_offset = 1e-10):
        '''
        Part two of a three part class computation.
        '''
        if np.any(self.weights != None) and not overwrite:
            raise OverwriteException('Barycentric coordinates already computed. Set flag overwrite if the computation should be updated.')
            
        # Ensure triangulation is available
        assert self.del_tri != None, 'Barycentric coordinates can not be computed. Triangulation is not available. First call method Triangulate().'
        # Ensure p_out is of the same dimensions
        assert np.ma.size(p_out, axis=1) == self.ndim, f'output points p_out are of dimension {np.ma.size(p_out, axis=1)} whereas the input points are of dimension {self.ndim}.'
        
        # Save the output-points
        self.p_out = p_out

        if self.timeit:
            t0 = time.time()
        # Compute the simplex_loc, i.e. in which triangulation (simplex) does each point lie
        self.simplex_loc = self.del_tri.find_simplex(p_out)

        if np.any(self.simplex_loc==-1):
            if special:
                something_changed = False
                # Attempt to correct if the offset is small enough
                points_outside = self.p_out[self.simplex_loc==-1, :]
                
                perturb_directions = np.array([[1, 0, 0],
                                               [-1, 0,0],
                                               [0,1,0],
                                               [0,-1,0],
                                               [0,0,1],
                                               [0,0,-1]])
                for p_idx, point_out in enumerate(points_outside):
                    for direction in perturb_directions:
                        p_new = point_out + direction*max_offset
                        print(f'{direction}')
                        if self.del_tri.find_simplex(p_new) >= 0:
                            points_outside[p_idx] = p_new
                            something_changed = True
                            print(f'{p_new}')
                            break
                
                if something_changed:
                    self.p_out[self.simplex_loc==-1, :] = points_outside
                    self.simplex_loc = self.del_tri.find_simplex(p_out)
                
            self.n_points_outside = np.sum(self.simplex_loc < 0)

            if (not batchmode) and (self.n_points_outside  > 0):
                warnings.warn(f'{self.n_points_outside :,} point(s) lie outside the Convex hull')

        # Extract the simplex indices (with ndim+1 points) corresponding to each output_point
        self.vertices = self.del_tri.simplices[self.simplex_loc]

        # Similarly get the transform for each corresponding simplex to compute barycentric coordinates
        transform = self.del_tri.transform[self.simplex_loc]

        # Compute the delta (see scipy documentation)
        delta = self.p_out - transform[:, self.ndim]

        # Compute the barymetric coordinates for each respective simplex using numpy's Einstein summation function
        self.bary = np.einsum('njk,nk->nj', transform[:, :self.ndim, :], delta)
        
        # Finally compute the weights for each vertex using the barymetric coordinates
        self.weights = np.hstack((self.bary, 1 - self.bary.sum(axis=1, keepdims=True)))
        
        if self.timeit:
            t1 = time.time()
        
        if self.show_done:
            print(f'Simplices corresponding to the given {np.ma.size(self.p_out, axis=0):,} points are found and the barycentric coordinates have been computed.')
            if self.timeit:
                print(f'Process took {np.round(t1-t0, 1)} seconds.')
        
        # End function
        pass
    
    def __call__(self, values):
        '''
        Part three of a three part class computation.
        '''
        assert np.ma.size(values, axis=0) == np.ma.size(self.p_in, axis=0), f'Input values are of size {np.shape(values)} whereas input points are of size {np.shape(self.p_in)}.'
        assert np.all(self.weights != None), 'Linear triangulation can not be performed. Barycentric coordinates are not available. First call method ComputeBarycentric().'
        
        if self.timeit:
            t0 = time.time()
        # Perform the Linear triangulation using numpy's Einstein summation function
        out = np.einsum('nj,nj->n', np.take(values, self.vertices), self.weights)
        if self.timeit:
            t1 = time.time()
            print(f'Process took {np.round((t1-t0)*1e6, 1)} microseconds.')
        
        # Fill points outside convex hull with nan
        out[np.any(self.weights<0, axis=1)] = np.nan
        
        # End function and return result
        return out

# Define functions for normal and tangential velocity components
def vel_normal(vel_uvw, normal_xyz):
    return ((np.sum(vel_uvw * normal_xyz, axis=1)).reshape((-1,1)) * normal_xyz)
def vel_tangent(vel_uvw, normal_xyz):
    return vel_uvw - vel_normal(vel_uvw, normal_xyz)
    
def load_unstructured_vtk_grid_clean(filepath, vtk_type_num, plotting=False):
    # First load dirty unstructured grid
    ug_dirty, _ = load_unstructured_vtk_grid(filepath, plotting=plotting)
    
    # Run clean-up procedure on
    ug_clean, adapted_ug_clean = vtkmesh_cleanup(ug_dirty, vtk_type_num)
    
    return ug_clean, adapted_ug_clean
    
def vtkmesh_cleanup(mesh, vtk_type_num):
    # Meshes need to be "cleaned up", because when exporting from Gmsh, there are cells of other dimensionality (e.g. vertex cells and line cells) added to the .vtk files.
    # These cause issues when later visualising in software such as Tecplot or Paraview.
    
    # Input: mesh which needs to be cleaned up,
    #        minimum number of of simplex which needs to remain,
    #        vtk type numbef of simplex which needs to remain (see 'https://examples.vtk.org/site/VTKFileFormats/')
    # Output: cleaned up mesh
    
    # Define new unstructured grid
    ug = vtk.vtkUnstructuredGrid()

    # Set points on the grid the same as the input mesh
    ug.SetPoints(mesh.GetPoints())

    # Define a cell array and fill this with the cells of desired cell_type (number of nodes)
    ca = vtk.vtkCellArray()
    mesh_cells = mesh.GetCells()
    mesh_conn_array = numpy_support.vtk_to_numpy(mesh_cells.GetConnectivityArray())
    for c_idx in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(c_idx)
        if cell.GetCellType() == vtk_type_num:
            cell_NoP = cell.GetNumberOfPoints()
            offset = mesh_cells.GetOffset(c_idx)
            cell_connectivity = mesh_conn_array[offset:offset+cell_NoP]
            ca.InsertNextCell(cell_NoP, cell_connectivity)

    ug.SetCells(vtk_type_num, ca)
    adapted_ug = dsa.WrapDataObject(ug)
    return ug, adapted_ug

# Load grid data
def load_unstructured_vtk_grid(filepath, plotting=False):
    # Input: filepath of unstructured vtk grid
    # Output: Unstructured vtk grid object
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()
    
    # Get output
    output = reader.GetOutput()
    # Wrap output in a dsa.WrapDataObject
    output_dsa = dsa.WrapDataObject(output)

    if plotting:
        scalar_range = output.GetScalarRange()

        # Create the mapper that corresponds the objects of the vtk file
        # into graphics elements
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(output)
        mapper.SetScalarRange(scalar_range)

        # Create the Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetOpacity(0.2)
        # Create the Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0, 0, 0) # Set background to black

        # Create the RendererWindow
        renderer_window = vtk.vtkRenderWindow()
        renderer_window.AddRenderer(renderer)

        # Create the RendererWindowInteractor and display the vtk_file
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renderer_window)
        interactor.Initialize()
        interactor.Start()
    
    return output, output_dsa

def load_csv_velocity(filepath, sep=' ', skiprows=4, index_col=False,
                      encoding='utf8', **kwargs):
    # Input: filepath of velocity field csv data
    # Output: pandas dataframe
    return pd.read_csv(filepath, sep=sep, skiprows=skiprows, encoding=encoding,
                       index_col=index_col, **kwargs)
    
def load_triangle_mesh(filepath):
    # HONESTLY CONSIDER THIS TO BE UNNECESSARY
    # Input: filepath of stl file
    # Output: Open3D TriangleMesh object
    return o3d.io.read_triangle_mesh(filepath)

def np_nodal_coordinates_from_unstructured_vtk_grid(unstructured_vtk_grid):
    # Input: unstructured_vtk_grid
    # Output: (N, 3)-Numpy array of grid coordinates
    return numpy_support.vtk_to_numpy(unstructured_vtk_grid.GetPoints().GetData())

def np_cell_coordinates_from_unstructured_vtk_grid(unstructured_vtk_grid):
    # Input: unstructured_vtk_grid
    # Output: (N, 3)-Numpy array of grid coordinates
    return numpy_support.vtk_to_numpy(unstructured_vtk_grid.GetCells().GetData())

def distance_from_points_to_o3dmesh(o3dmesh, points, scene=None, output_scene=False):
    # Input: open3d trianglemesh, points as (N, 3)-Numpy array (and scene if desired)
    # Output: (N,)-Numpy array of distances from each point (and scene if desired)
    
    # Create a RayCastingScene from the stl mesh
    if scene == None:
        scene = o3d.t.geometry.RaycastingScene()
        trianglemesh = o3d.t.geometry.TriangleMesh.from_legacy(o3dmesh)
        _ = scene.add_triangles(trianglemesh)
    
    # Output the distance
    o3d_distances = scene.compute_distance(points.astype(np.float32))
    
    # Return the distances as numpy array
    if output_scene:
        return o3d_distances.numpy(), scene
    else:
        return o3d_distances.numpy()

def pointnormals_from_unstructured_vtk(unstructured_vtk):
    '''Compute PointNormals from an unstructured vtk grid.
    In: vtkUnstructuredGrid
    Out: (N, 3)-Numpy array with N=number of points in the grid'''
    # Transform the unstructured grid to type PolyData
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(unstructured_vtk)
    geometry_filter.Update()
    polydata = geometry_filter.GetOutput()

    # Use this PolyData to compute the normals on cells
    polydatanormals = vtk.vtkPolyDataNormals()
    polydatanormals.SetInputData(polydata)
    polydatanormals.ComputePointNormalsOff()
    polydatanormals.ComputeCellNormalsOn()
    polydatanormals.Update()
    
    # Extract the Normals array at the PointData level
    return np.array(polydatanormals.GetOutput().GetCellData().GetNormals())

# Interpolator function
def interpolate_velocity_uvw_with_RBF(coordinates_xyz, velocity_uvw, nodes_xyz, RBFmethod='thin-plate-spline', neighbors=100, timing=False, **kwargs):
    # Input:
    # Output: (N, 3)-numpy array of velocities interpolated at the nodes_xyz
    
    # Initialise the uvw array
    uvw = np.zeros_like(nodes_xyz)
    
    # Interpolate the u-velocity
    if timing:
        print(f'RBF interpolating u-velocity with {RBFmethod}...')
        t0 = time.time()
    uvw[:,0] = sp.interpolate.RBFInterpolator(coordinates_xyz, velocity_uvw[:,0], neighbors=neighbors, kernel=RBFmethod, **kwargs)(nodes_xyz)
    if timing:
        t1 = time.time()
        print(f'DONE after {round(t1-t0, 3)} seconds')
    
    # Interpolate the v-velocity
    if timing:
        print(f'RBF interpolating v-velocity with {RBFmethod}...')
        t0 = time.time()
    uvw[:,1] = sp.interpolate.RBFInterpolator(coordinates_xyz, velocity_uvw[:,1], neighbors=neighbors, kernel=RBFmethod, **kwargs)(nodes_xyz)
    if timing:
        t1 = time.time()
        print(f'DONE after {round(t1-t0, 3)} seconds')
        
    # Interpolate the w-velocity
    if timing:
        print(f'RBF interpolating w-velocity with {RBFmethod}...')
        t0 = time.time()
    uvw[:,2] = sp.interpolate.RBFInterpolator(coordinates_xyz, velocity_uvw[:,2], neighbors=neighbors, kernel=RBFmethod, **kwargs)(nodes_xyz)
    if timing:
        t1 = time.time()
        print(f'DONE after {round(t1-t0, 3)} seconds')
    
    return uvw

def compute_volume_results():
    return None

def compute_surface_results(coordinates_xyz, normals_xyz,
                            velfield_xyz, velfield_uvw,
                            distances,
                            RBFmethod='thin-plate-spline', neighbors=100,
                            time_interpolation=False, convert_mm_to_m=False,
                            **kwargs):
    # Input: All of the above
    # Output: A numpy data array of the following (see below)
    
    # Initialise data array
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
    # 10: interpolated u-velocity in-plane
    # 11: interpolated v-velocity in-plane
    # 12: interpolated w-velocity in-plane
    # 13: interpolated absolute velocity in-plane
    # 14: interpolated u-velocity out-of-plane
    # 15: interpolated v-velocity out-of-plane
    # 16: interpolated w-velocity out-of-plane
    # 17: interpolated absolute velocity in-plane
    # 18: interpolated shearing u-velocity in-plane
    # 19: interpolated shearing v-velocity in-plane
    # 20: interpolated shearing w-velocity in-plane
    # 21: interpolated shearing absolute velocity in-plane
    data = np.zeros((len(coordinates_xyz[:,0]),22))
    
    # Assign the coordinates to the data array
    data[:,:3] = coordinates_xyz
    
    # Assign the normals to the data array
    data[:,3:6] = normals_xyz
    
    # Compute and assign the interpolated velocities
    if convert_mm_to_m:
        data[:, 6:9] = interpolate_velocity_uvw_with_RBF(velfield_xyz/1000,
                                                         velfield_uvw,
                                                         coordinates_xyz/1000,
                                                         RBFmethod=RBFmethod, neighbors=neighbors, timing=time_interpolation, **kwargs)
    else:
        data[:, 6:9] = interpolate_velocity_uvw_with_RBF(velfield_xyz,
                                                         velfield_uvw,
                                                         coordinates_xyz,
                                                         RBFmethod=RBFmethod, neighbors=neighbors, timing=time_interpolation, **kwargs)
        
    
    # Compute velocity magnitude
    data[:,9] = np.linalg.norm(data[:,6:9], axis=1)
    
    # Compute in-plane velocities
    data[:,10:13] = vel_tangent(data[:,6:9], data[:,3:6])
    data[:,13] = np.linalg.norm(data[:,10:13], axis=1)
    
    # Compute out-of-plane velocities
    data[:,14:17] = vel_normal(data[:,6:9], data[:,3:6])
    data[:,17] = np.linalg.norm(data[:,14:17], axis=1)
    
    # Compute shearing velocities
    data[:,18:22] = data[:,10:14] / distances.reshape((-1,1))
    
    return data
    
def append_volume_data(vtk_wrapped_data_object, data, identifier=''):
    # Input:
    # Output:
        
    # Define the identifier text
    if identifier!='':
        identifier_text = ' - ' + identifier
    else:
        identifier_text = ''
    
    vtk_wrapped_data_object.PointData.append(data[:,6],  f'Velocity U{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,7],  f'Velocity V{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,8],  f'Velocity W{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,9],  f'Velocity{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,10],  'Model distance')
    vtk_wrapped_data_object.PointData.append(data[:,27], 'IsValid')
    
def append_skinFriction_data(vtk_wrapped_data_object, data, add_normals=True, identifier=''):    
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
    
    # Define the identifier text
    if identifier!='':
        identifier_text = ' - ' + identifier
    else:
        identifier_text = ''
        
    # Start with the normals if desired
    if add_normals:
        vtk_wrapped_data_object.CellData.append(data[:,3], 'Normal - X')
        vtk_wrapped_data_object.CellData.append(data[:,4], 'Normal - Y')
        vtk_wrapped_data_object.CellData.append(data[:,5], 'Normal - Z')
    
    vtk_wrapped_data_object.CellData.append(data[:,6],  f'Velocity U{identifier_text} [m/s]')
    vtk_wrapped_data_object.CellData.append(data[:,7],  f'Velocity V{identifier_text} [m/s]')
    vtk_wrapped_data_object.CellData.append(data[:,8],  f'Velocity W{identifier_text} [m/s]')
    vtk_wrapped_data_object.CellData.append(data[:,9],  f'Velocity{identifier_text} [m/s]')
    vtk_wrapped_data_object.CellData.append(data[:,10], f'Wall-Normal Velocity Gradient In-Plane U-component{identifier_text} [Hz]')
    vtk_wrapped_data_object.CellData.append(data[:,11], f'Wall-Normal Velocity Gradient In-Plane V-component{identifier_text} [Hz]')
    vtk_wrapped_data_object.CellData.append(data[:,12], f'Wall-Normal Velocity Gradient In-Plane W-component{identifier_text} [Hz]')
    vtk_wrapped_data_object.CellData.append(data[:,13], f'Wall-Normal Velocity Gradient In-Plane Magnitude{identifier_text} [Hz]')
    vtk_wrapped_data_object.CellData.append(data[:,14], f'IsValid{identifier_text} [-]')
    vtk_wrapped_data_object.CellData.append(data[:,15], f'Local ground-shift{identifier_text} [mm]')
    vtk_wrapped_data_object.CellData.append(data[:,16], f'Plane-Fitted ground-shift{identifier_text} [mm]')
    vtk_wrapped_data_object.CellData.append(data[:,17], f'Ground shift isComputed{identifier_text} [-]')
    vtk_wrapped_data_object.CellData.append(data[:,18], f'Thresholded B.L. Height{identifier_text} [mm]')
    vtk_wrapped_data_object.CellData.append(data[:,19], f'B.L. IsValid{identifier_text} [-]')
    vtk_wrapped_data_object.CellData.append(data[:,20], f'# of FluidMesh points{identifier_text} [-]')
    vtk_wrapped_data_object.CellData.append(data[:,21], f'# of ObjectMesh points{identifier_text} [-]')
    
    
    # Finally append skin friction data to the normals on the cell
    # vectors = vtk.vtkDoubleArray()
    # vectors.SetNumberOfTuples(len(data[:,0]))
    # vectors.SetNumberOfComponents(3)
    # for c_idx in range(len(data[:,0])):
    #     vectors.InsertComponent(c_idx, 0, data[c_idx,10])
    #     vectors.InsertComponent(c_idx, 1, data[c_idx,11])
    #     vectors.InsertComponent(c_idx, 2, data[c_idx,12])
    
    # vtk_wrapped_data_object.CellData.VTKObject.SetVectors(vectors)
    
def append_surface_data(vtk_wrapped_data_object, data, add_normals=True, identifier=''):
    # Input:
    # Output:
    
    # Define the identifier text
    if identifier!='':
        identifier_text = ' - ' + identifier
    else:
        identifier_text = ''
        
    # Start with the normals if desired
    if add_normals:
        vtk_wrapped_data_object.PointData.append(data[:,3], 'Normal - X')
        vtk_wrapped_data_object.PointData.append(data[:,4], 'Normal - Y')
        vtk_wrapped_data_object.PointData.append(data[:,5], 'Normal - Z')
    
    vtk_wrapped_data_object.PointData.append(data[:,6],  f'Velocity U{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,7],  f'Velocity V{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,8],  f'Velocity W{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,9],  f'Velocity{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,10],  'Surface distance')
    vtk_wrapped_data_object.PointData.append(data[:,11], f'Velocity U - in-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,12], f'Velocity V - in-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,13], f'Velocity W - in-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,14], f'Velocity - in-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,15], f'Velocity U - out-of-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,16], f'Velocity V - out-of-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,17], f'Velocity W - out-of-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,18], f'Velocity - out-of-plane{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,19], f'Velocity U - shearing{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,20], f'Velocity V - shearing{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,21], f'Velocity W - shearing{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,22], f'Velocity - shearing{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,23], f'Velocity U - wall-projection{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,24], f'Velocity V - wall-projection{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,25], f'Velocity W - wall-projection{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,26], f'Velocity - wall-projection{identifier_text}')
    vtk_wrapped_data_object.PointData.append(data[:,27], 'IsValid')
    
    # No need to return anything
def save_adapted_vtk_dataset_as_unstructured_grid(vtk_wrapped_data_object, savefilepath):
    # Input: 
    # Output: No output, this functions only saves the vtkObject
    
    # Create a vtk writer for the unstructured grid
    writer = vtk.vtkXMLUnstructuredGridWriter()
    # Set the filename
    writer.SetFileName(savefilepath)
    # Define the inputdata
    writer.SetInputData(vtk_wrapped_data_object.VTKObject)
    # Write the file
    writer.Write()
    
    # No need to return anything
    
def saveNumpyStyle(np_array, savefilepath):
    np.save(savefilepath, np_array)
    print(f'File saved as:\n{savefilepath}')
    # No need to return anything
    
def scanRow(row, zoneInfoBlock, dataSeparator=' '):
    # Initialize dictionaries
    generalVariables = {}
    zoneInfo = {}
    
    searchData = re.search(r'^-*[0-9\.]+', row)
    searchVariables = re.search(r'(?i)\bvariables\b', row)

    if searchVariables:
        variablesString = row.split('=')[1]
        if re.search('",\s"', row):
            dataSeparator = '", "'
        else:
            dataSeparator = '" "'
        variablesListWithUnits = variablesString.lstrip(' "').lstrip('"').rstrip('"\n').split(dataSeparator)
        variablesList = [item.split('[')[0].rstrip(' ') for item in variablesListWithUnits]
    else:
        variablesList = []
    
    # Check if there is a zone in there
    zoneInfoLine = re.search(r'(?i)\bzone\b', row)
    
    if (zoneInfoLine or zoneInfoBlock) and (not searchData):
        hits = re.findall(r'([\w]+)\s?=\s?([\w\s"]+)',
                          row,
                          flags=re.IGNORECASE
                          )

        for hitPair in hits:
            # Extract key and value from hitPair
            key = hitPair[0].capitalize()
            value = hitPair[1].strip('"')
            
            # Save the zone-specific information
            zoneInfo[key] = value
    else:
        hits = re.findall(r'([\w]+)\s?=\s?([\w\s"]+)',
                          row,
                          flags=re.IGNORECASE
                          )
        
        for hitPair in hits:
            # Extract key and value from hitPair
            key = hitPair[0].capitalize()
            value = hitPair[1].strip('"')
            
            generalVariables[key] = value

    return searchData, (searchVariables, variablesList), generalVariables, (zoneInfoLine, zoneInfo)
    
def fileQuickScan(filename, encoding='utf8', maxToCheckNumberOfLines=100, variablesSeparator=' '):
    generalVariables = {}
    zoneInfoPD = pd.DataFrame()
    # 1) open the file in binary mode to quickly determine number of rows
    with open(filename, mode = 'rb') as f:
        numberOfRows = sum(1 for i in f)
        
    # 2) Now open the file to read the header data
    # Initialise variables
    fileLoop = True
    rowStart = 0
    
    while fileLoop:
        zoneInfoLocalPD = pd.DataFrame()
        zoneInfoBlock = False
        if rowStart >= numberOfRows:
            break
        with open(filename, mode='r', encoding=encoding) as f:
            for i, row in enumerate(islice(f, rowStart, None)):
                rowUse = row.rstrip('\n')
                # Scan the row for info in the top lines
                searchData, searchVariables, generalVar, zoneInfo = scanRow(rowUse,
                                                                            zoneInfoBlock,
                                                                            dataSeparator=variablesSeparator)
                
                if searchVariables[0]:
                    variablesList = searchVariables[1]
                elif (zoneInfo[0] or zoneInfoBlock):
                    zoneInfoBlock = True
                    for key in zoneInfo[1].keys():
                        zoneInfoLocalPD[key] = np.array([zoneInfo[1][key]])
                elif generalVar:
                    for key in generalVar.keys():
                        generalVariables[key] = generalVar[key]
                
                if searchData:
                    # First check how many data-rows can be expected
                    numberOfDataRows = int(zoneInfoLocalPD['I']) * int(zoneInfoLocalPD['J']) * int(zoneInfoLocalPD['K'])
                    zoneInfoLocalPD['Np'] = numberOfDataRows
                    
                    # Then save the number of header lines to skip
                    startRowData = i + rowStart
                    zoneInfoLocalPD['startRowData'] = startRowData
                    rowStart = startRowData + numberOfDataRows
                    
                    # Add the local Zone info for the corresponding time frame
                    zoneInfoPD = pd.concat([zoneInfoPD, zoneInfoLocalPD],
                                           ignore_index=True)
                    
                    break
                
                # "Fail-safe"
                if i > maxToCheckNumberOfLines:
                    numberOfLinesToSkip = None
                    fileLoop = False
                    break
        
        # if zoneInfoLine:
        #     zoneData = pd.concat([zoneData, zoneDataRow], ignore_index=True)
            
    return (numberOfRows, variablesList,
            generalVariables, zoneInfoPD)
