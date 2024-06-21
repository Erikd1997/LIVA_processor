# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:26:22 2023

Main Python file for applying methods to obtain skin friction
assimilation

@author: ErikD
"""

# Change the working directory to the one the file is run from
# (cause of damn spyder)
import os
import time
# import warnings
cwdir = os.path.abspath('')
os.chdir(cwdir)

# import multiprocessing as mp
import tecplot as tp
import numpy as np
# import pandas as pd
import scipy as sp
from pathlib import Path
import vtk
from vtk.util import numpy_support
# from vtk.numpy_interface import dataset_adapter as dsa
# import matplotlib.pyplot as plt
# import open3d as o3d

# import own modules
# import mp_workers as mpw
import pivDataAssimilationSurfaceParticlesFraction_SkinFriction as pivDA
import tecplot_format as tpf

# pipe vtk output errors to file
errOut = vtk.vtkFileOutputWindow()
errOut.SetFileName("VTK Error Out.txt")
vtkStdErrOut = vtk.vtkOutputWindow()
vtkStdErrOut.SetInstance(errOut)

# =============================================================================
# Define which methods to run and method-specific settings
# =============================================================================

# Some general parameters of the experiment (and binned data)
RADIUS_px = 16 # px
grid_spacing_mm = 3.0666421270904727
grid_spacing_px = 8
percentageTrackingPartitions = 100

#######################################################
######## Jux' method
method0 = False
test_method0 = False
savemethod0 = False
method0_FindBL = False
method0_cstart = 0
RADIUS_griddata = round(grid_spacing_mm * 2.45, 2) # A little over 2
method0_savename_addition = '_methodJux-Extrapolated'
# method0_savename_addition = '_method0-modified-interp-10mm_offset'

#######################################################
######## Method 1 - Bin-Con-LinInterp
method1 = False
test_method1 = False
savemethod1 = False
method1_FindBL = False
method1_cstart = 0
RADIUS_griddata = round(grid_spacing_mm * 2.45, 2) # A little over 2
method1_savename_addition = '_methodBin-Con-LinInterp'

#######################################################
######## Method 2 - Bin-Nudge-QuadFit
method2 = False
test_method2 = False
savemethod2 = False
method2_FindBL = False
method2_cstart = 0
# Settings of method2
divfree = False
mm_per_px = grid_spacing_mm  / grid_spacing_px
RADIUS2 = RADIUS_px * mm_per_px * 1.1
grid_size = grid_spacing_mm
ORDER = 2 # order of the least squares
method2_savename_addition = '_methodBin-Nudge-QuadFit-SunFlower-HighSpeed'

#######################################################
######## Method 3 - Tracer-Nudge-QuadFit
method3 = False
test_method3 = False
savemethod3 = False
method3_FindBL = False
method3_cstart = 0
method3_savename_addition = '_methodTracer-Nudge-QuadFit'
RADIUS3 = 7.0 # [mm]

#######################################################
######## Method 4 - Tracer-Con-QuadFit
method4 = False
test_method4 = False
savemethod4 = False
method4_FindBL = False
method4_cstart = 0
method4_savename_addition = '_methodTracer-Con-QuadFit'
RADIUS4 = 7.0 # [mm]

#######################################################
######## Method 5 - Coin-Stacked-GroundTruth
methodGT = True
test_methodGT = False
savemethodGT = True
methodGT_FindBL = False
methodGT_cstart = 0
RADIUSGT_COIN = 12.0 # [mm]
coinHeight = 3.0     # [mm]
coinOverlap = 50    # [%]
methodGT_savename_addition = '_methodTracer-GroundTruth-H3mm-75per'
coinFitMethod = 'LIN'

#######################################################
######## General settings
show_in_tecplot = False
performFullRun = True
xmin = -150; xmax = 150
ymin = -150; ymax = 150
zmin = 0; zmax = 150

output_mesh_to_compute = 'surface'

start_text = f'Script is run with following main settings.\nOutputmesh-type: {output_mesh_to_compute}\nShow results in TecPlot: {show_in_tecplot}'
start_text += f'\nRun method0: {method0}'
if method0:
    start_text += f'\n\tSave method0: {savemethod0}'
start_text += f'\nRun method1: {method1}'
if method1:
    start_text += f'\n\tSave method1: {savemethod1}'
start_text += f'\nRun method2: {method2}'
if method2:
    start_text += f'\n\tSave method2: {savemethod2}'
    start_text += f'\n\tSphere Radius: {RADIUS2:.3f} mm'
start_text += f'\nRun method3: {method3}'
if method3:
    start_text += f'\n\tSave method3: {savemethod3}'
    start_text += f'\n\tSphere Radius: {RADIUS3:.3f} mm'
start_text += f'\nRun method4: {method4}'
if method4:
    start_text += f'\n\tSave method4: {savemethod4}'
    start_text += f'\n\tSphere Radius: {RADIUS4:.3f} mm'
start_text += f'\nRun method GroundTruth: {methodGT}'
if methodGT:
    start_text += f'\n\tSave method GroundTruth: {savemethodGT}'
    start_text += f'\n\tCoin Radius: {RADIUSGT_COIN:.3f} mm'
    start_text += f'\n\tCoin Height: {coinHeight:.3f} mm'
    start_text += f'\n\tCoin Overlap: {coinOverlap} %'
    start_text += f'\n\tCoin fitting method: {coinFitMethod }'

print(start_text)
# =============================================================================
# Load grids and velocity data
# =============================================================================
# Define the path of the mesh files
data_dir = Path(r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Erik_cube_binning_32x32x32')
testdata_dir = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data")
dirpath_meshes = Path(r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes')
dirpath_savefolder = Path(r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Results')
planeFitFolder = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\GroundPlaneFits"
# Defining a mesh of the object's surface, used to define the surface coordinates \/
# filepath_surfacemesh = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Offset_WBJ\WBJ_with_ground_no_offset.vtk")
# filepath_surfacemesh = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Offset_cyclist\Cyclist_with_ground.vtk")

# Used for defining the distance to the nearest surface
# filepath_contourstl = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Offset_WBJ\WBJ_with_ground_no_offset.stl")

# Used for masking points inside body \/
filepath_objectstl = data_dir / 'Cube_12cm.stl'
# filepath_objectstl = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Offset_WBJ\WBJ.stl")
# filepath_objectstl = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Offset_cyclist\Cyclist.stl")

## Filepath for the outputmeshes

## Offset meshes for 8px, 6px, 4px and 2px:
############################# CUBE Outputs #############################
########################################################################
# Surface meshes
filepath_outputmesh = dirpath_meshes / r'cube\cube-surface-similar_mesh_size_as_binning2.vtk'
# filepath_outputmesh = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes\cube\cube-surface-7mm_spacing.vtk")
# filepath_outputmesh = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes\cube\cube-surface-14mm_spacing.vtk")
# filepath_outputmesh = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshes\cube\cube-surface-6_747mm_spacing.vtk")
print(f'Output mesh: {filepath_outputmesh.name}\n')

# Define the path of the velocity field data (binned and tracers)
filepath_velfield_binned = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Bins\Cube_binning_32x32x32_75per0001.dat")
# filepath_velfield_tracks = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax200_ymin-200_ymax200_zmin-20_zmax200.npy")
filepath_velfield_tracks = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax200_ymin-200_ymax200_zmin-20_zmax200-Filter1.npy")

#%%
# =============================================================================
# Main code
# =============================================================================

# Import velocity field data as pandas dataframes
columnHeaders = ["x", "y", "z", "u", "v", "w", "omegax", "omegay", "omegaz", "isValid"]
if method0 or method1 or method2:
    velfield_binned = pivDA.load_csv_velocity(filepath_velfield_binned,
                                       names=columnHeaders,
                                       usecols=[0, 1, 2, 3, 4, 5, 20, 21, 22, 51])
    # velfield_binned = pivDA.load_csv_velocity(filepath_velfield_binned, names=["x", "y", "z", "u", "v", "w", "isValid"])
velfield_tracks = np.load(filepath_velfield_tracks, allow_pickle=True)
velfield_tracks_full = np.vstack(velfield_tracks)

gridIsLoaded = False
def loadGrids():
    global unstructured_vtk_outputmesh, adapted_unstructured_vtk_outputmesh, Ncells
    
    print('Loading in mesh data...')
    # Import vtk grids
    unstructured_vtk_outputmesh_dirty, _ = pivDA.load_unstructured_vtk_grid(filepath_outputmesh.as_posix())
    
    # Clean up meshes
    unstructured_vtk_outputmesh, adapted_unstructured_vtk_outputmesh = pivDA.vtkmesh_cleanup(unstructured_vtk_outputmesh_dirty, vtk.VTK_TRIANGLE)
         
    # Extract the unstructured grid nodal points as numpy array
    Ncells = unstructured_vtk_outputmesh.GetNumberOfCells()
    # np_output_points = pivDA.np_nodal_coordinates_from_unstructured_vtk_grid(unstructured_vtk_outputmesh)
    
    # Extract the unstructured grid nodal cells as numpy array
    # np_output_cells = unstructured_vtk_outputmesh.GetCells()

def extractNormalsAndDistancesFromMesh():
    global np_distances_output, scene, np_normals_output
    print('Started extracting data from mesh\n.\n.')
    # =============================================================================
    # Extract necessary data from the mesh
    # =============================================================================
    print(f'Total number of cells in the output mesh {Ncells:,}')
    
    # Compute distances
    # np_distances_output = np.ones((np.ma.size(np_output_points, axis=0),))
    
    # Compute normal vectors
    np_normals_output = pivDA.pointnormals_from_unstructured_vtk(unstructured_vtk_outputmesh)

#%%
print('Starting methods\n.\n.')
if method0:
    print('########## Executing METHOD-0 ##########')
    if not gridIsLoaded: 
        loadGrids()
        extractNormalsAndDistancesFromMesh()
        gridIsLoaded = True
    # =============================================================================
    # Manipulate the mesh and add data to it
    # METHOD 0 - Only introduce object mesh, do not impose no-slip condition
    #         Triangulation with linear interpolation
    #               without imposing physics
    #               information from model
    # =============================================================================
    ### Structure of results dataset
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
    output_method0 = np.zeros((Ncells,20))
    
    # Assign the normals to the data array
    output_method0[:,3:6] = np_normals_output
    
    # Initialize an instance of the CompactSphereSupportedApproximation_VTK class
    CSSA0 = pivDA.CompactSphereSupportedApproximation_VTK(RADIUS_griddata,
                                                         ORDER,
                                                         griddata_mode=True,
                                                         updateGroundEstimate=True,
                                                         useObjectInfo=False,
                                                         velMethod='0')
    
    # Add the output mesh
    CSSA0.AddOutputMesh_VTK(unstructured_vtk_outputmesh)
    
    # Add model polydata
    STLreader = vtk.vtkSTLReader()
    STLreader.SetFileName(filepath_objectstl.as_posix())
    STLreader.Update()
    CSSA0.AddModelPolyData_VTK(STLreader.GetOutput())
    
    ## Interpolate velocities using own implementation of griddata with compact supported spheres.
    ####################################################################################
    ###### Define velfield for ground-plane estimate
    ## Note that the inputted "groundPlane" tracers must retain tracer
    ## particles inside the object + ground
    
    # This function retains tracers inside the ground + model
    mask_method0_groundPlaneTracers_inneredge = pivDA.maskWithOffsetToSTL(velfield_tracks_full[:,:3],
                                                                          filepath_objectstl.as_posix(),
                                                                          offset = 5)
    
    # This function removes tracers inside the ground + model
    mask_method0_groundPlaneTracers_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                            filepath_objectstl.as_posix(),
                                                                            offset = 10)
    
    mask_method0_groundPlaneTracers = (mask_method0_groundPlaneTracers_inneredge == 1) & (mask_method0_groundPlaneTracers_outeredge==0)
    groundPlaneTracers_banded_method0 = velfield_tracks_full[mask_method0_groundPlaneTracers, :3]
    
    CSSA0.AddGroundPlaneTracers(groundPlaneTracers_banded_method0)
    
    ####################################################################################
    ###### Define velfield for interpolation
    
    # Define inner and outer edges to crop the velocity field and improve performance
    mask_method0_velfield_inneredge = pivDA.maskWithOffsetToSTL(velfield_binned.to_numpy()[:,:3],
                                                                filepath_objectstl.as_posix(),
                                                                offset = 5)
    
    mask_method0_velfield_outeredge = pivDA.mask_by_offset_to_stl(velfield_binned.to_numpy()[:,:3],
                                                                  filepath_objectstl.as_posix(),
                                                                  offset = 50)
    # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
    _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(velfield_binned.to_numpy()[:,:3], STLreader.GetOutput())

    # Mask the binned velocity field
    mask_method0_velfield = ((mask_method0_velfield_inneredge == 1) &
                             (mask_method0_velfield_outeredge == 0) &
                             (maskInsideSTL == 0) &
                             (velfield_binned['isValid'])
                             )
    velfield_banded_method0 = velfield_binned.to_numpy()[mask_method0_velfield, :]
    
    # Add velocity field
    CSSA0.AddVelocityField_Numpy(velfield_banded_method0, columns=columnHeaders)
    
    # "Build" the fluid mesh
    CSSA0.BuildFluidMeshFromVelocityField()
    # Build the pointlocators
    CSSA0.BuildPointLocators()
    # Detrend ground-plane
    CSSA0.ComputeGroundPlaneFit(dz_ground=5, Rvary=12, Rlocal=None, minDistanceToObject=12,
                               track_percentage=True, partitions=100,
                               recompute=False, savefolder=planeFitFolder)
    print('.')
    print('.')
    # Start the iteration
    if performFullRun:
        CSSA0.Run(track_percentage=True, partitions=percentageTrackingPartitions, c_start=method0_cstart,
                 dz_ground=5, maxNStepsAlongNormal=100, test=test_method0,
                 findBL=method0_FindBL)
        
        # Assign the coordinates to the data array
        output_method0[:,:3] = CSSA0.xyz
    
        # Save velocities
        output_method0[:,6] = CSSA0.u
        output_method0[:,7] = CSSA0.v
        output_method0[:,8] = CSSA0.w
        
        # Compute velocity magnitude in output mesh
        output_method0[:,9] = np.linalg.norm(output_method0[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output_method0[:,10:13] = CSSA0.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output_method0[:,13] = np.linalg.norm(CSSA0.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output_method0[:,14] = CSSA0.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output_method0[:,15] = CSSA0.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output_method0[:,16] = CSSA0.shiftedPoints[:,2] - CSSA0.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output_method0[:,17] = CSSA0.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output_method0[:,18] = CSSA0.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output_method0[:,19] = CSSA0.BLvalid.astype(int)
        
        # Save the number of fluid and object mesh nodes
        output_method0[:,20] = CSSA0.info[:,0]
        output_method0[:,21] = CSSA0.info[:,1]
                
        # Swap all points where data is NaN with explicit "0"
        output_method0_mod = output_method0.copy()
        output_method0_mod[~CSSA0.isValid,6:-1] = np.zeros_like(output_method0_mod[0,6:-1])
        
        if savemethod0:
            # =============================================================================
            # Append the results to the adapatble grid object and save
            # =============================================================================
            # Append the results    
            pivDA.append_skinFriction_data(adapted_unstructured_vtk_outputmesh, output_method0_mod, identifier='method_0')
            
            # Save the results
            savefilename = Path(filepath_outputmesh.as_posix().replace('.vtk', method0_savename_addition+'.vtu')).name
            savefilepath_output_method0 = (dirpath_savefolder / savefilename).as_posix()
            savefilepath_output_method0_numpy = savefilepath_output_method0.replace('.vtu', '.npy')
            
            # As a fail-safe, first save the results as a numpy array
            pivDA.saveNumpyStyle(output_method0_mod, savefilepath_output_method0_numpy)
            
            pivDA.save_adapted_vtk_dataset_as_unstructured_grid(adapted_unstructured_vtk_outputmesh, savefilepath_output_method0)
            print(f'Saved method-0 as\n{savefilepath_output_method0}\n\n')        
        else:
            print('Method-0 was not saved...')
    else:
        print('Method-0 was only initialised...')
else:
    savefilepath_output_method0 = filepath_outputmesh.as_posix().replace('.vtk', method0_savename_addition+'.vtu')

# =============================================================================
# # Method 1: Modify the data with no-slip condition using linear triangulation
# =============================================================================
#%%
if method1:
    print('########## Executing METHOD-1 ##########')
    if not gridIsLoaded: 
        loadGrids()
        extractNormalsAndDistancesFromMesh()
        gridIsLoaded = True
    ### Structure of results dataset
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
    # Initialise array
    output_method1 = np.zeros((Ncells,22))
        
    # Assign the normals to the data array
    output_method1[:,3:6] = np_normals_output
    
    # Initialize an instance of the CompactSphereSupportedApproximation_VTK class
    CSSA1 = pivDA.CompactSphereSupportedApproximation_VTK(RADIUS_griddata,
                                                         ORDER,
                                                         griddata_mode=True,
                                                         updateGroundEstimate=True,
                                                         useObjectInfo=True,
                                                         velMethod='1')

    # Add the output mesh
    CSSA1.AddOutputMesh_VTK(unstructured_vtk_outputmesh)
    
    # Add model polydata
    STLreader = vtk.vtkSTLReader()
    STLreader.SetFileName(filepath_objectstl.as_posix())
    STLreader.Update()
    CSSA1.AddModelPolyData_VTK(STLreader.GetOutput())
    
    ####################################################################################
    ##### Add the ground-plane tracers to adjust the object registration
    ## Note that the inputted "groundPlane" tracers must retain tracer
    ## particles inside the object + ground
    
    # This function retains tracers inside the ground + model
    mask_method1_groundPlaneTracers_inneredge = pivDA.maskWithOffsetToSTL(velfield_tracks_full[:,:3],
                                                                          filepath_objectstl.as_posix(),
                                                                          offset = 5)
    
    # This function removes tracers inside the ground + model
    mask_method1_groundPlaneTracers_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                            filepath_objectstl.as_posix(),
                                                                            offset = 10)
    
    mask_method1_groundPlaneTracers = (mask_method1_groundPlaneTracers_inneredge == 1) & (mask_method1_groundPlaneTracers_outeredge==0)
    groundPlaneTracers_banded_method1 = velfield_tracks_full[mask_method1_groundPlaneTracers, :3]
    
    CSSA1.AddGroundPlaneTracers(groundPlaneTracers_banded_method1)
    
    ####################################################################################
    ##### Add the binned velocity field
    # Define the banded velocity field
    mask_method1_velfield_inneredge = pivDA.maskWithOffsetToSTL(velfield_binned.to_numpy()[:,:3],
                                                         filepath_objectstl.as_posix(),
                                                         offset = 5)

    mask_method1_velfield_outeredge = pivDA.mask_by_offset_to_stl(velfield_binned.to_numpy()[:,:3],
                                                         filepath_objectstl.as_posix(),
                                                         offset = 50)

    # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
    _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(velfield_binned.to_numpy()[:,:3], STLreader.GetOutput())

    # Mask the binned velocity field
    mask_method1_velfield = ((mask_method1_velfield_inneredge == 1) &
                             (mask_method1_velfield_outeredge == 0) &
                             (maskInsideSTL == 0) &
                             (velfield_binned['isValid'])
                             )
    velfield_use_method1 = velfield_binned.to_numpy()[mask_method1_velfield, :]
    
    
    # Add velocity field
    CSSA1.AddVelocityField_Numpy(velfield_use_method1, columns=columnHeaders)
    
    # "Build" the fluid mesh
    CSSA1.BuildFluidMeshFromVelocityField()
    # Build the pointlocators
    CSSA1.BuildPointLocators()
    # Add slicer
    CSSA1.AddSlicer('plane')
    
    # Detrend ground-plane
    CSSA1.ComputeGroundPlaneFit(dz_ground=5, Rvary=12, Rlocal=None, minDistanceToObject=12,
                               track_percentage=True, partitions=100,
                               recompute=False, savefolder=planeFitFolder)
    print('.')
    print('.')
    # Start the iteration
    if performFullRun:
        CSSA1.Run(track_percentage=True, partitions=percentageTrackingPartitions,
                 dz_ground=5, maxNStepsAlongNormal=400, c_start=method1_cstart,
                 test=test_method1, findBL=method1_FindBL)
        # Assign the coordinates to the data array
        output_method1[:,:3] = CSSA1.xyz
    
        # Save velocities
        output_method1[:,6] = CSSA1.u
        output_method1[:,7] = CSSA1.v
        output_method1[:,8] = CSSA1.w
    
        # Compute velocity magnitude in output mesh
        output_method1[:,9] = np.linalg.norm(output_method1[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output_method1[:,10:13] = CSSA1.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output_method1[:,13] = np.linalg.norm(CSSA1.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output_method1[:,14] = CSSA1.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output_method1[:,15] = CSSA1.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output_method1[:,16] = CSSA1.shiftedPoints[:,2] - CSSA1.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output_method1[:,17] = CSSA1.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output_method1[:,18] = CSSA1.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output_method1[:,19] = CSSA1.BLvalid.astype(int)
        
        # Save the number of fluid and object mesh nodes
        output_method1[:,20] = CSSA1.info[:,0]
        output_method1[:,21] = CSSA1.info[:,1]
    
        # Swap all points where data is NaN with explicit "0"
        output_method1_mod = output_method1.copy()
        output_method1_mod[~CSSA1.isValid,6:-1] = np.zeros_like(output_method1_mod[0,6:-1])
        
        if savemethod1:
            # =============================================================================
            # Append the results to the adapatble grid object and save
            # =============================================================================
            # Save the results
            savefilename = Path(filepath_outputmesh.as_posix().replace('.vtk', method1_savename_addition+'.vtu')).name
            savefilepath_output_method1 = (dirpath_savefolder / savefilename).as_posix()
            savefilepath_output_method1_numpy = savefilepath_output_method1.replace('.vtu', '.npy')
            
            # As a fail-safe, first save the results as a numpy array
            pivDA.saveNumpyStyle(output_method1_mod, savefilepath_output_method1_numpy)
            
            # Append the results
            pivDA.append_skinFriction_data(adapted_unstructured_vtk_outputmesh, output_method1_mod, identifier='method_1')
            
            pivDA.save_adapted_vtk_dataset_as_unstructured_grid(adapted_unstructured_vtk_outputmesh, savefilepath_output_method1)
            print(f'Saved method-1 as\n{savefilepath_output_method1}\n\n')
        else:
            print('Method-1 was not saved...')
    else:
        print('Method-1 was only initialised...')
else:
    savefilepath_output_method1 = filepath_outputmesh.as_posix().replace('.vtk', method1_savename_addition+'.vtu')
        
    
# =====================================================================
#                         Method 2:
#         Second-order wall condition regression in
#           spherical interrogation volume from binned data
# =====================================================================
#%%
if method2:
    print('########## Executing METHOD-2 ##########')
    loadGrids()
    extractNormalsAndDistancesFromMesh()
    ### Structure of results dataset
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
    
    # Initialise array
    output_method2 = np.zeros((Ncells,22))
        
    # Assign the normals to the data array
    output_method2[:,3:6] = np_normals_output
    
    # Initialize an instance of the CompactSphereSupportedApproximation_VTK class
    CSSA2 = pivDA.CompactSphereSupportedApproximation_VTK(RADIUS2,
                                                         ORDER,
                                                         griddata_mode=False,
                                                         updateGroundEstimate=True,
                                                         useObjectInfo=True,
                                                         velMethod='2')

    

    # Add the output mesh
    CSSA2.AddOutputMesh_VTK(unstructured_vtk_outputmesh)
    
    # Add model polydata
    STLreader = vtk.vtkSTLReader()
    STLreader.SetFileName(filepath_objectstl.as_posix())
    STLreader.Update()
    CSSA2.AddModelPolyData_VTK(STLreader.GetOutput())
    
    ####################################################################################
    ##### Add the ground-plane tracers to adjust the object registration
    ## Note that the inputted "groundPlane" tracers must retain tracer
    ## particles inside the object + ground
    
    # This function retains tracers inside the ground + model
    mask_method2_groundPlaneTracers_inneredge = pivDA.maskWithOffsetToSTL(velfield_tracks_full[:,:3],
                                                                          filepath_objectstl.as_posix(),
                                                                          offset = 5)
    
    # This function removes tracers inside the ground + model
    mask_method2_groundPlaneTracers_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                            filepath_objectstl.as_posix(),
                                                                            offset = 10)
    
    mask_method2_groundPlaneTracers = (mask_method2_groundPlaneTracers_inneredge == 1) & (mask_method2_groundPlaneTracers_outeredge==0)
    groundPlaneTracers_banded_method2 = velfield_tracks_full[mask_method2_groundPlaneTracers, :3]
    
    CSSA2.AddGroundPlaneTracers(groundPlaneTracers_banded_method2)
    
    ####################################################################################
    ##### Add the binned velocity field
    # Define the banded velocity field
    mask_method2_velfield_inneredge = pivDA.maskWithOffsetToSTL(velfield_binned.to_numpy()[:,:3],
                                                                filepath_objectstl.as_posix(),
                                                                offset = 5)

    mask_method2_velfield_outeredge = pivDA.mask_by_offset_to_stl(velfield_binned.to_numpy()[:,:3],
                                                                  filepath_objectstl.as_posix(),
                                                                  offset = 50)

    # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
    _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(velfield_binned.to_numpy()[:,:3], STLreader.GetOutput())

    # Mask the binned velocity field
    mask_method2_velfield = ((mask_method2_velfield_inneredge == 1) &
                             (mask_method2_velfield_outeredge == 0) &
                             (maskInsideSTL == 0) &
                             (velfield_binned['isValid'])
                             )
    velfield_use_method2 = velfield_binned.to_numpy()[mask_method2_velfield, :]
    
    # Add velocity field
    CSSA2.AddVelocityField_Numpy(velfield_use_method2, columns=columnHeaders)
    
    # # Determine the particle concentration
    # mask_p_in_domain = ((velfield_binned.iloc[:,0] >= xmin) & (velfield_binned.iloc[:,0] <= xmax) &
    #                     (velfield_binned.iloc[:,1] >= ymin) & (velfield_binned.iloc[:,1] <= ymax) &
    #                     (velfield_binned.iloc[:,2] >= zmin) & (velfield_binned.iloc[:,2] <= zmax) &
    #                     (maskInsideSTL == 0)
    #                     )
    # NumberOfParticles = np.sum(mask_p_in_domain)
    
    # # Define filter to obtain domain volume
    # volumeSTLFilter = vtk.vtkMassProperties()
    # volumeSTLFilter.SetInputConnection(STLreader.GetOutputPort())
    # volumeSTLFilter.Update()

    # volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) - volumeSTLFilter.GetVolume()
    # ratio = NumberOfParticles / volume
    concentration = 1/grid_size**2
    
    CSSA2.AddConcentration(concentration)
    
    print(f'Using particle concentration of {round(concentration, 4)} particles per mm^{3}')
    
    # "Build" the fluid mesh
    CSSA2.BuildFluidMeshFromVelocityField()
    # Build the pointlocators
    CSSA2.BuildPointLocators()
    # Detrend ground-plane
    CSSA2.ComputeGroundPlaneFit(dz_ground=5, Rvary=12, Rlocal=None, minDistanceToObject=12,
                               track_percentage=True, partitions=100,
                               recompute=False, savefolder=planeFitFolder)
    print('.')
    print('.')
    
    # Start the iteration
    if performFullRun:
        CSSA2.Run(track_percentage=True, partitions=percentageTrackingPartitions,
                 dz_ground=5, maxNStepsAlongNormal=400, c_start=method2_cstart,
                 test=test_method2, findBL=method2_FindBL)
        
        # Save velocities
        output_method2[:,6] = CSSA2.u
        output_method2[:,7] = CSSA2.v
        output_method2[:,8] = CSSA2.w
    
        # Compute velocity magnitude in output mesh
        output_method2[:,9] = np.linalg.norm(output_method2[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output_method2[:,10:13] = CSSA2.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output_method2[:,13] = np.linalg.norm(CSSA2.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output_method2[:,14] = CSSA2.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output_method2[:,15] = CSSA2.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output_method2[:,16] = CSSA2.shiftedPoints[:,2] - CSSA2.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output_method2[:,17] = CSSA2.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output_method2[:,18] = CSSA2.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output_method2[:,19] = CSSA2.BLvalid.astype(int)
        
        # Save the number of fluid and object mesh nodes
        output_method2[:,20] = CSSA2.info[:,0]
        output_method2[:,21] = CSSA2.info[:,1]
    
        # Swap all points where data is NaN with explicit "0"
        output_method2_mod = output_method2.copy()
        output_method2_mod[~CSSA2.isValid,6:-1] = np.zeros_like(output_method2_mod[0,6:-1])
    
        # =============================================================================
        # Append the results to the adaptable grid object and save
        # =============================================================================
        if savemethod2:
            # =============================================================================
            # Append the results to the adapatble grid object and save
            # =============================================================================
            # Save the results
            savefilename = Path(filepath_outputmesh.as_posix().replace('.vtk', method2_savename_addition+'.vtu')).name
            savefilepath_output_method2 = (dirpath_savefolder / savefilename).as_posix()
            savefilepath_output_method2_numpy = savefilepath_output_method2.replace('.vtu', '.npy')
            
            # As a fail-safe, first save the results as a numpy array
            pivDA.saveNumpyStyle(output_method2_mod, savefilepath_output_method2_numpy)
            
            # Append the results    
            pivDA.append_skinFriction_data(adapted_unstructured_vtk_outputmesh, output_method2_mod, identifier='method_2')
            
            pivDA.save_adapted_vtk_dataset_as_unstructured_grid(adapted_unstructured_vtk_outputmesh, savefilepath_output_method2)
            print(f'Saved method-2 as\n{savefilepath_output_method2}\n\n')
        else:
            print('Method-2 was not saved...')
    else:
        print('Method-2 was only initialised...')
else:
    savefilepath_output_method2 = filepath_outputmesh.as_posix().replace('.vtk', method2_savename_addition+'.vtu')
    
    
# =====================================================================
#                         Method 3:
#         Second-order wall condition regression in
#           spherical interrogation volume from tracer data
# =====================================================================
#%%
if method3:
    print('########## Executing METHOD-3 ##########')
    if not gridIsLoaded: 
        loadGrids()
        extractNormalsAndDistancesFromMesh()
        gridIsLoaded = True
    ### Structure of results dataset
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
    
    # Initialise array
    output_method3 = np.zeros((Ncells,22))
        
    # Assign the normals to the data array
    output_method3[:,3:6] = np_normals_output
    
    # Initialize an instance of the CompactSphereSupportedApproximation_VTK class
    CSSA3 = pivDA.CompactSphereSupportedApproximation_VTK(RADIUS3,
                                                         ORDER,
                                                         griddata_mode=False,
                                                         updateGroundEstimate=True,
                                                         useObjectInfo=True,
                                                         velMethod='3')

    

    # Add the output mesh
    CSSA3.AddOutputMesh_VTK(unstructured_vtk_outputmesh)
    
    # Add model polydata
    STLreader = vtk.vtkSTLReader()
    STLreader.SetFileName(filepath_objectstl.as_posix())
    STLreader.Update()
    CSSA3.AddModelPolyData_VTK(STLreader.GetOutput())
    
    ####################################################################################
    ##### Add the ground-plane tracers to adjust the object registration
    ## Note that the inputted "groundPlane" tracers must retain tracer
    ## particles inside the object + ground
    
    # This function retains tracers inside the ground + model
    mask_method3_groundPlaneTracers_inneredge = pivDA.maskWithOffsetToSTL(velfield_tracks_full[:,:3],
                                                                          filepath_objectstl.as_posix(),
                                                                          offset = 5)
    
    # This function removes tracers inside the ground + model
    mask_method3_groundPlaneTracers_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                            filepath_objectstl.as_posix(),
                                                                            offset = 10)
    
    mask_method3_groundPlaneTracers = (mask_method3_groundPlaneTracers_inneredge == 1) & (mask_method3_groundPlaneTracers_outeredge==0)
    groundPlaneTracers_banded_method3 = velfield_tracks_full[mask_method3_groundPlaneTracers, :3]
    
    CSSA3.AddGroundPlaneTracers(groundPlaneTracers_banded_method3)
    
    ####################################################################################
    ##### Add the binned velocity field
    # Define the banded velocity field

    mask_method3_velfield_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                  filepath_objectstl.as_posix(),
                                                                  offset = 10)

    # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
    _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(velfield_tracks_full[:,:3], STLreader.GetOutput())

    # Mask the binned velocity field
    mask_method3_velfield = ((mask_method3_groundPlaneTracers_inneredge == 1) &
                             (mask_method3_velfield_outeredge == 0) &
                             (maskInsideSTL == 0)
                             )
    velfield_use_method3 = velfield_tracks_full[mask_method3_velfield, :]
    
    # Add velocity field
    columnHeadersTracers = ['X [mm]', 'Y [mm]', 'Z [mm]', 'U [m/s]', 'V [m/s]',
                            'W [m/s]', 'Timestep [-]', 'Track ID [-]', 
                            'Number of Cameras [-]'
                            ]
    CSSA3.AddVelocityField_Numpy(velfield_use_method3, columns=columnHeadersTracers)
    
    # Determine the particle concentration
    mask_p_in_domain = ((velfield_tracks_full[:,0] >= xmin) & (velfield_tracks_full[:,0] <= xmax) &
                        (velfield_tracks_full[:,1] >= ymin) & (velfield_tracks_full[:,1] <= ymax) &
                        (velfield_tracks_full[:,2] >= zmin) & (velfield_tracks_full[:,2] <= zmax) &
                        (maskInsideSTL == 0)
                        )
    p_in_domain = velfield_tracks_full[mask_p_in_domain, :3]
    NumberOfParticles = len(p_in_domain)
    
    # Define filter to obtain domain volume
    volumeSTLFilter = vtk.vtkMassProperties()
    volumeSTLFilter.SetInputConnection(STLreader.GetOutputPort())
    volumeSTLFilter.Update()

    volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) - volumeSTLFilter.GetVolume()
    ratio = volume / NumberOfParticles
    mean_distance = (3/(4*np.pi*ratio))**(1/3)
    concentration = 1 / mean_distance**2
    CSSA3.AddConcentration(concentration)
    
    # "Build" the fluid mesh
    CSSA3.BuildFluidMeshFromVelocityField()
    # Build the pointlocators
    CSSA3.BuildPointLocators()
    # Detrend ground-plane
    CSSA3.ComputeGroundPlaneFit(dz_ground=5, Rvary=12, Rlocal=None, minDistanceToObject=12,
                               track_percentage=True, partitions=100,
                               recompute=False, savefolder=planeFitFolder)
    print('.')
    print('.')
    # Start the iteration
    if performFullRun:
        CSSA3.Run(track_percentage=True, partitions=percentageTrackingPartitions,
                 dz_ground=5, maxNStepsAlongNormal=400, c_start=method3_cstart,
                 test=test_method3, findBL=method3_FindBL)
        
        # Save velocities
        output_method3[:,6] = CSSA3.u
        output_method3[:,7] = CSSA3.v
        output_method3[:,8] = CSSA3.w
    
        # Compute velocity magnitude in output mesh
        output_method3[:,9] = np.linalg.norm(output_method3[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output_method3[:,10:13] = CSSA3.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output_method3[:,13] = np.linalg.norm(CSSA3.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output_method3[:,14] = CSSA3.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output_method3[:,15] = CSSA3.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output_method3[:,16] = CSSA3.shiftedPoints[:,2] - CSSA3.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output_method3[:,17] = CSSA3.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output_method3[:,18] = CSSA3.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output_method3[:,19] = CSSA3.BLvalid.astype(int)
    
        # Save the number of fluid and object mesh nodes
        output_method3[:,20] = CSSA3.info[:,0]
        output_method3[:,21] = CSSA3.info[:,1]
        
        # Swap all points where data is NaN with explicit "0"
        output_method3_mod = output_method3.copy()
        output_method3_mod[~CSSA3.isValid,6:-1] = np.zeros_like(output_method3_mod[0,6:-1])
    
        # =============================================================================
        # Append the results to the adaptable grid object and save
        # =============================================================================
        if savemethod3:
            # =============================================================================
            # Append the results to the adapatble grid object and save
            # =============================================================================
            # Save the results
            savefilename = Path(filepath_outputmesh.as_posix().replace('.vtk', method3_savename_addition+'.vtu')).name
            savefilepath_output_method3 = (dirpath_savefolder / savefilename).as_posix()
            savefilepath_output_method3_numpy = savefilepath_output_method3.replace('.vtu', '.npy')
            
            # As a fail-safe, first save the results as a numpy array
            pivDA.saveNumpyStyle(output_method3_mod, savefilepath_output_method3_numpy)
            
            # Append the results    
            pivDA.append_skinFriction_data(adapted_unstructured_vtk_outputmesh, output_method3_mod, identifier='method_3')
            
            pivDA.save_adapted_vtk_dataset_as_unstructured_grid(adapted_unstructured_vtk_outputmesh, savefilepath_output_method3)
            print(f'Saved method-3 as\n{savefilepath_output_method3}\n\n')
        else:
            print('Method-3 was not saved...')
    else:
        print('Method-3 was only initialised...')
else:
    savefilepath_output_method3 = filepath_outputmesh.as_posix().replace('.vtk', method3_savename_addition+'.vtu')

# =====================================================================
#                         Method 4:
#         Second-order wall condition regression in
#           spherical interrogation volume from tracer
#         data with single constraint at projection point
# =====================================================================
#%%
if method4:
    print('########## Executing METHOD-4 ##########')
    if not gridIsLoaded: 
        loadGrids()
        extractNormalsAndDistancesFromMesh()
        gridIsLoaded = True
    ### Structure of results dataset
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
    
    # Initialise array
    output_method4 = np.zeros((Ncells,22))
        
    # Assign the normals to the data array
    output_method4[:,3:6] = np_normals_output
    
    # Initialize an instance of the CompactSphereSupportedApproximation_VTK class
    CSSA4 = pivDA.CompactSphereSupportedApproximation_VTK(RADIUS4,
                                                         ORDER,
                                                         griddata_mode=False,
                                                         updateGroundEstimate=True,
                                                         useObjectInfo=True,
                                                         LSR_constrained=True,
                                                         velMethod='4')

    # Add the output mesh
    CSSA4.AddOutputMesh_VTK(unstructured_vtk_outputmesh)
    
    # Add model polydata
    STLreader = vtk.vtkSTLReader()
    STLreader.SetFileName(filepath_objectstl.as_posix())
    STLreader.Update()
    CSSA4.AddModelPolyData_VTK(STLreader.GetOutput())
    
    ####################################################################################
    ##### Add the ground-plane tracers to adjust the object registration
    ## Note that the inputted "groundPlane" tracers must retain tracer
    ## particles inside the object + ground
    
    # This function retains tracers inside the ground + model
    mask_method4_groundPlaneTracers_inneredge = pivDA.maskWithOffsetToSTL(velfield_tracks_full[:,:3],
                                                                          filepath_objectstl.as_posix(),
                                                                          offset = 5)
    
    # This function removes tracers inside the ground + model
    mask_method4_groundPlaneTracers_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                            filepath_objectstl.as_posix(),
                                                                            offset = 10)
    
    mask_method4_groundPlaneTracers = (mask_method4_groundPlaneTracers_inneredge == 1) & (mask_method4_groundPlaneTracers_outeredge==0)
    groundPlaneTracers_banded_method4 = velfield_tracks_full[mask_method4_groundPlaneTracers, :3]
    
    CSSA4.AddGroundPlaneTracers(groundPlaneTracers_banded_method4)
    
    ####################################################################################
    ##### Add the binned velocity field
    # Define the banded velocity field

    mask_method4_velfield_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                  filepath_objectstl.as_posix(),
                                                                  offset = 10)

    # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
    _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(velfield_tracks_full[:,:3], STLreader.GetOutput())

    # Mask the binned velocity field
    mask_method4_velfield = ((mask_method4_groundPlaneTracers_inneredge == 1) &
                             (mask_method4_velfield_outeredge == 0) &
                             (maskInsideSTL == 0)
                             )
    velfield_use_method4 = velfield_tracks_full[mask_method4_velfield, :]
    
    # Add velocity field
    columnHeadersTracers = ['X [mm]', 'Y [mm]', 'Z [mm]', 'U [m/s]', 'V [m/s]',
                            'W [m/s]', 'Timestep [-]', 'Track ID [-]', 
                            'Number of Cameras [-]'
                            ]
    CSSA4.AddVelocityField_Numpy(velfield_use_method4, columns=columnHeadersTracers)
    
    # "Build" the fluid mesh
    CSSA4.BuildFluidMeshFromVelocityField()
    # Build the pointlocators
    CSSA4.BuildPointLocators()
    # Detrend ground-plane
    CSSA4.ComputeGroundPlaneFit(dz_ground=5, Rvary=12, Rlocal=None, minDistanceToObject=12,
                               track_percentage=True, partitions=100,
                               recompute=False, savefolder=planeFitFolder)
    print('.')
    print('.')
    
    # Start the iteration
    if performFullRun:
        CSSA4.Run(track_percentage=True, partitions=percentageTrackingPartitions,
                 dz_ground=5, maxNStepsAlongNormal=400, c_start=method4_cstart,
                 test=test_method4, findBL=method4_FindBL)
        
        # Save velocities
        output_method4[:,6] = CSSA4.u
        output_method4[:,7] = CSSA4.v
        output_method4[:,8] = CSSA4.w
    
        # Compute velocity magnitude in output mesh
        output_method4[:,9] = np.linalg.norm(output_method4[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output_method4[:,10:13] = CSSA4.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output_method4[:,13] = np.linalg.norm(CSSA4.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output_method4[:,14] = CSSA4.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output_method4[:,15] = CSSA4.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output_method4[:,16] = CSSA4.shiftedPoints[:,2] - CSSA4.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output_method4[:,17] = CSSA4.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output_method4[:,18] = CSSA4.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output_method4[:,19] = CSSA4.BLvalid.astype(int)
    
        # Save the number of fluid and object mesh nodes
        output_method4[:,20] = CSSA4.info[:,0]
        output_method4[:,21] = CSSA4.info[:,1]
        
        # Swap all points where data is NaN with explicit "0"
        output_method4_mod = output_method4.copy()
        output_method4_mod[~CSSA4.isValid,6:-1] = np.zeros_like(output_method4_mod[0,6:-1])
    
        # =============================================================================
        # Append the results to the adaptable grid object and save
        # =============================================================================
        if savemethod4:
            # =============================================================================
            # Append the results to the adapatble grid object and save
            # =============================================================================
            # Save the results
            savefilename = Path(filepath_outputmesh.as_posix().replace('.vtk', method4_savename_addition+'.vtu')).name
            savefilepath_output_method4 = (dirpath_savefolder / savefilename).as_posix()
            savefilepath_output_method4_numpy = savefilepath_output_method4.replace('.vtu', '.npy')
            
            # As a fail-safe, first save the results as a numpy array
            pivDA.saveNumpyStyle(output_method4_mod, savefilepath_output_method4_numpy)
            
            # Append the results    
            pivDA.append_skinFriction_data(adapted_unstructured_vtk_outputmesh, output_method4_mod, identifier='method_4')
            
            pivDA.save_adapted_vtk_dataset_as_unstructured_grid(adapted_unstructured_vtk_outputmesh, savefilepath_output_method4)
            print(f'Saved method-4 as\n{savefilepath_output_method4}\n\n')
        else:
            print('Method-4 was not saved...')
    else:
        print('Method-4 was only initialised...')
else:
    savefilepath_output_method4 = filepath_outputmesh.as_posix().replace('.vtk', method4_savename_addition+'.vtu')

# =====================================================================
#                         Method Ground-Truth:
#          An approach to the ground truth using very thin coins
#              which are stacked on top of each other
#          spherical interrogation volume from tracer data
# =====================================================================
#%%
if methodGT:
    print('########## Executing METHOD-GroundTruth ##########')
    if not gridIsLoaded: 
        loadGrids()
        extractNormalsAndDistancesFromMesh()
        gridIsLoaded = True
    ### Structure of results dataset
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
    
    # CHECKED
    # Initialise array
    output_methodGT = np.zeros((Ncells,22))

    # CHECKED
    # Assign the normals to the data array
    output_methodGT[:,3:6] = np_normals_output

    # CHECKED
    # Initialize an instance of the CompactSphereSupportedApproximation_VTK class
    CSSAGT = pivDA.CompactSphereSupportedApproximation_VTK(RADIUSGT_COIN,
                                                         ORDER,
                                                         griddata_mode=False,
                                                         updateGroundEstimate=True,
                                                         useObjectInfo=True,
                                                         LSR_constrained=False,
                                                         VOItype='c',
                                                         coinHeight=coinHeight,
                                                         coinOverlap=coinOverlap,
                                                         fitMethod=coinFitMethod,
                                                         velMethod='GT')

    # CHECKED
    # Add the output mesh
    CSSAGT.AddOutputMesh_VTK(unstructured_vtk_outputmesh)
    
    # CHECKED
    # Add model polydata (# Is this even used anymore?)
    STLreader = vtk.vtkSTLReader()
    STLreader.SetFileName(filepath_objectstl.as_posix())
    STLreader.Update()
    CSSAGT.AddModelPolyData_VTK(STLreader.GetOutput())
    
    ####################################################################################
    ##### Add the ground-plane tracers to adjust the object registration
    ## Note that the inputted "groundPlane" tracers must retain tracer
    ## particles inside the object + ground
    # CHECKED
    # This function retains tracers inside the ground + model
    mask_methodGT_groundPlaneTracers_inneredge = pivDA.maskWithOffsetToSTL(velfield_tracks_full[:,:3],
                                                                          filepath_objectstl.as_posix(),
                                                                          offset = 5)
    
    # This function removes tracers inside the ground + model
    mask_methodGT_groundPlaneTracers_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                            filepath_objectstl.as_posix(),
                                                                            offset = 10)
    
    mask_methodGT_groundPlaneTracers = (mask_methodGT_groundPlaneTracers_inneredge == 1) & (mask_methodGT_groundPlaneTracers_outeredge==0)
    groundPlaneTracers_banded_methodGT = velfield_tracks_full[mask_methodGT_groundPlaneTracers, :3]
    
    CSSAGT.AddGroundPlaneTracers(groundPlaneTracers_banded_methodGT)
    
    ####################################################################################
    ##### Add the binned velocity field
    # Define the banded velocity field
    # CHECKED
    mask_methodGT_velfield_outeredge = pivDA.mask_by_offset_to_stl(velfield_tracks_full[:,:3],
                                                                  filepath_objectstl.as_posix(),
                                                                  offset = 4*coinHeight
                                                                  )
    # CHECKED
    # Add mask to bin all points inside the object (I do not update the groundlocation on the object)
    _, _, maskInsideSTL = pivDA.maskCoordinatesInsideVTKPD(velfield_tracks_full[:,:3], STLreader.GetOutput())

    # Mask the binned velocity field
    mask_methodGT_velfield = ((mask_methodGT_groundPlaneTracers_inneredge == 1) &
                             (mask_methodGT_velfield_outeredge == 0) &
                             (maskInsideSTL == 0)
                             )
    velfield_use_methodGT = velfield_tracks_full[mask_methodGT_velfield, :]
    
    # CHECKED
    # Add velocity field
    columnHeadersTracers = ['X [mm]', 'Y [mm]', 'Z [mm]', 'U [m/s]', 'V [m/s]',
                            'W [m/s]', 'Timestep [-]', 'Track ID [-]', 
                            'Number of Cameras [-]'
                            ]
    CSSAGT.AddVelocityField_Numpy(velfield_use_methodGT, columns=columnHeadersTracers)
    
    # CHECKED
    # "Build" the fluid mesh
    CSSAGT.BuildFluidMeshFromVelocityField()
    # CHECKED
    # Build the pointlocators
    CSSAGT.BuildPointLocators()
    # CHECKED
    # Detrend ground-plane
    CSSAGT.ComputeGroundPlaneFit(dz_ground=5, Rvary=12, Rlocal=None, minDistanceToObject=12,
                                 track_percentage=True, partitions=100,
                                 recompute=False, savefolder=planeFitFolder)
    print('.')
    print('.')
    
    # CHECKED
    # Start the iteration
    if performFullRun:
        CSSAGT.Run(track_percentage=True, partitions=percentageTrackingPartitions,
                 dz_ground=5, maxNStepsAlongNormal=400, c_start=methodGT_cstart,
                 test=test_methodGT, findBL=methodGT_FindBL)
        
        # CHECKED
        # Save velocities
        output_methodGT[:,6] = CSSAGT.u
        output_methodGT[:,7] = CSSAGT.v
        output_methodGT[:,8] = CSSAGT.w
    
        # Compute velocity magnitude in output mesh
        output_methodGT[:,9] = np.linalg.norm(output_methodGT[:,6:9], axis=1)
        
        # Save the Wall-normal gradients in-plane
        output_methodGT[:,10:13] = CSSAGT.velWallNormalGradientInPlane
        
        # Compute the wall-normal gradients in-plane magnitude
        output_methodGT[:,13] = np.linalg.norm(CSSAGT.velWallNormalGradientInPlane, axis=1)
        
        # Save the velocity computation is valid flag
        output_methodGT[:,14] = CSSAGT.isValid.astype(int)
        
        # Save the local wall-normal ground shift
        output_methodGT[:,15] = CSSAGT.shiftVector[:,2]
        
        # Save the fitted wall-normal ground shift
        output_methodGT[:,16] = CSSAGT.shiftedPoints[:,2] - CSSAGT.xyz[:,2]
        
        # Save the "GroundShiftIsComputed" flag
        output_methodGT[:,17] = CSSAGT.shiftComputed.astype(int)
        
        # Save the thresholded boundary layer height
        output_methodGT[:,18] = CSSAGT.delta
        
        # Save the "BoundaryLayerIsValid" flag
        output_methodGT[:,19] = CSSAGT.BLvalid.astype(int)
    
        # Save the number of fluid and object mesh nodes
        output_methodGT[:,20] = CSSAGT.info[:,0]
        output_methodGT[:,21] = CSSAGT.info[:,1]
        
        # Swap all points where data is NaN with explicit "0"
        output_methodGT_mod = output_methodGT.copy()
        output_methodGT_mod[~CSSAGT.isValid,6:-1] = np.zeros_like(output_methodGT_mod[0,6:-1])
    
        # =============================================================================
        # Append the results to the adaptable grid object and save
        # =============================================================================
        if savemethodGT:
            # =============================================================================
            # Append the results to the adapatble grid object and save
            # =============================================================================
            # Save the results
            savefilename = Path(filepath_outputmesh.as_posix().replace('.vtk', methodGT_savename_addition+'.vtu')).name
            savefilepath_output_methodGT = (dirpath_savefolder / savefilename).as_posix()
            savefilepath_output_methodGT_numpy = savefilepath_output_methodGT.replace('.vtu', '.npy')
            
            # As a fail-safe, first save the results as a numpy array
            pivDA.saveNumpyStyle(output_methodGT_mod, savefilepath_output_methodGT_numpy)
            
            # Append the results    
            pivDA.append_skinFriction_data(adapted_unstructured_vtk_outputmesh, output_methodGT_mod, identifier='method_GT')
            
            pivDA.save_adapted_vtk_dataset_as_unstructured_grid(adapted_unstructured_vtk_outputmesh, savefilepath_output_methodGT)
            print(f'Saved method-GT as\n{savefilepath_output_methodGT}\n\n')
        else:
            print('Method-GT was not saved...')
    else:
        print('Method-GT was only initialised...')
else:
    savefilepath_output_methodGT = filepath_outputmesh.as_posix().replace('.vtk', methodGT_savename_addition+'.vtu')

#%%
# =============================================================================
# # Show results in tecplot
# =============================================================================
if show_in_tecplot:
    # =============================================================================
    # # Show all available results in tecplot
    # =============================================================================
    results = []
    if savemethod0:
        results.append(savefilepath_output_method0)
        
    if savemethod1:
        results.append(savefilepath_output_method1)
        
    if savemethod2:
        results.append(savefilepath_output_method2)
        
    if savemethod3:
        results.append(savefilepath_output_method3)
        
    if savemethod4:
        results.append(savefilepath_output_method4)
        
    if savemethodGT:
        results.append(savefilepath_output_methodGT)
        
    if len(results) > 0:
        # Initialise a pytecplot connection
        tp.session.connect()
        tp.new_layout()
        
        tp.active_page().name='Comparison CSSA vs full grid'
        
        # with tp.session.suspend():
        # Show each result
        for result_file in results:
            is_initial = result_file==results[0]
            frame = tpf.add_frame(tp.active_page(), result_file, initial=is_initial)
            
            # Add BL to the frame
    
        if len(results) > 1:
            tpf.format_frames(tp.active_page())
            
        tp.macro.execute_command('$!RedrawAll')
            