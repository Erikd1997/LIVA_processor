# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 08:15:09 2024

coinstackingFit_main

@author: ErikD
"""
import os
# import warnings
cwdir = os.path.abspath('')
os.chdir(cwdir)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import vtk
from skspatial.objects import Plane, Points
# from copy import deepcopy

import pivDataAssimilation as pivDA
from coinstackingFit import CoinStackSupportedApproximation_VTK

# Define data directories
data_dir = Path(r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Erik_cube_binning_32x32x32')
dirpath_meshes = Path(r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\gmsh-4.11.1-Windows64\test-saves')

# Define data files
filepath_surfacemesh = dirpath_meshes / 'cube-surface-nooffset_similar_mesh_size_as_binning.vtk'
filepath_objectstl = data_dir / 'Cube_12cm.stl'
filepath_outputmesh = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\gmsh-4.11.1-Windows64\test-saves\cube-volume.vtk")


# filepath_velfield_untouched_cube_tracks = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\Cube_5000dt_tracks-tuple-crop_xmin0_xmax200_ymin-60_ymax60_zmin-10_zmax80.npy")
# xmin = -200, xmax = 0 | ymin = -60, ymax = 60 | zmin = -20, zmax = 80
# filepath_velfield_untouched_cube_tracks  = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax0_ymin-60_ymax60_zmin-20_zmax80.npy"
# xmin = -60, xmax = 60 | ymin = -60, ymax = 60 | zmin = 100, zmax = 160
# filepath_velfield_untouched_cube_tracks = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-60_xmax60_ymin-60_ymax60_zmin100_zmax160.npy"
# xmin = 0, xmax = 200 | ymin = -60, ymax = 60 | zmin = -10, zmax = 80
# filepath_velfield_untouched_cube_tracks = r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin0_xmax200_ymin-60_ymax60_zmin-20_zmax80.npy"

filepath_velfield_untouched_cube_tracks  = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\Cubes\Tracks\x+-200y+-200z-50+200-tuple-crop_xmin-200_xmax200_ymin-200_ymax200_zmin-20_zmax200-Filter1.npy")

groundplane_path = Path(r"C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Test data\GroundPlaneFits\Ntracers=3838611_Ncells=46814_dz=5_Rvary=12_minDist=12_Rlocal=None.npy")

filedir = Path(r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\Thesis figures\Velocity reconstruction')

#%%
# Load velfield tracks
velfield = np.vstack(np.load(filepath_velfield_untouched_cube_tracks, allow_pickle=True))

# Load ground plane info
gp = np.load(groundplane_path, allow_pickle=True)

# Get information on the ground plane fit
originalPoints = gp[:, :3]
shiftedPoints = gp[:, 7:10] + originalPoints
shiftComputed = gp[:, 3].astype(bool)

## Recompute the fit since it is not saved:
shiftedPoints = shiftedPoints[shiftComputed, :]
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

# Define two points (X1, Y1) and (X2, Y2) which will lie on the plane
find_z = lambda x, y: (-d - a*x - b*y) / c

#%%
# Define coin stacking fit scheme
meanMode = False
use_plane_fit = False

# Define evaluation points
coinDirection = 'z'

x = -60
xmin = 90
xmax = 120
Nx = 1

y = -110

save_velfield = False
savefilepath = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\meshPipeline\numpy-saves\velfield-thin-coin_x='+f'{xmin}.npy'

savefig = False
savefile = r'C:\Users\ErikD\Documents\Me-Shizzle\Study\Master Courses\MasterThesis\src\figures\results\coin_stack_at_x='+f'{xmin}.png'

# z_ground = -3.4832825
# z_ground = 0
z_ground = find_z(xmin, y)
dz = z_ground - 2

z = 35
zmin = 0    
zmax = 30
zstep = 0.5
Nz = round((zmax-zmin) / zstep + 1)

# Define coin parameters
coinRadius = 12
coinOverlap = 75
# coinHeight = 1
# Nz = (zmax-zmin) / coinHeight

if coinDirection=='x':
    coinHeight = np.abs((xmax-xmin) / (Nx-1))
    xrange = np.linspace(xmin + coinHeight/2, xmax - coinHeight/2, Nx-1)
    zrange = np.linspace(zmin, zmax, Nz)
    
    xxrange, zzrange = np.meshgrid(xrange, z)
    points_to_eval = np.c_[xxrange.flatten(order='F'), y * np.ones_like(xxrange.flatten()), zzrange.flatten(order='F')]
    
    NperLine = Nx-1
    Nlines = Nz
    ground = xmax
    
elif coinDirection=='y':
    
    coinHeight = np.abs((ymax-ymin) / (Ny-1))
    # xrange = np.linspace(xmin, xmax, Nx)
    zrange = np.linspace(zmin + coinHeight/2, zmax - coinHeight/2, Nz-1)
    
    xxrange, zzrange = np.meshgrid(x, zrange)
    points_to_eval = np.c_[xxrange.flatten(order='F'), np.zeros_like(xxrange.flatten()), zzrange.flatten(order='F')]

    NperLine = Nz-1
    Nlines = Nx
else:
    xrange = np.linspace(xmin, xmax, Nx)
    zrange = np.linspace(zmin, zmax, Nz)
    coinHeight = zstep / (1 - coinOverlap/100)
    
    # xxrange, zzrange = np.meshgrid(x, zrange)
    xxrange, zzrange = np.meshgrid(xrange, zrange)
    points_to_eval = np.c_[xxrange.flatten(order='F'), y * np.ones_like(xxrange.flatten()), zzrange.flatten(order='F')]
    
    
    NperLine = Nz
    Nlines = Nx
    ground = zmin
    

# Create the CoinStacker object
coiner = CoinStackSupportedApproximation_VTK(coinRadius, coinHeight)

# Add evaluation points
coiner.AddOutputMesh_Numpy(points_to_eval, ground)

# Add velocity field
_, _, mask = pivDA.compute_points_inside_stl_and_ground(velfield[:,:3], filepath_objectstl.as_posix(), z_offset=dz)
velfield_use = velfield[(mask==0), :]
velfield_use[:,2] -= z_ground
# del velfield
# del velfield_cube_tracks
# mask = pivDA.mask_by_offset_to_stl(velfield.to_numpy()[:,:3],
#                                            filepath_objectstl.as_posix(),
#                                            offset = 0)
# velfield_use = velfield.to_numpy()[(mask) & (velfield['isValid']==1), :]
coiner.AddVelocityField_Numpy(velfield_use[:,:6])

# Add object mesh
reader = vtk.vtkSTLReader()
reader.SetFileName(filepath_objectstl.as_posix())
reader.Update()
coiner.AddModelPolyData_VTK(reader.GetOutput())

# Build the fluid mesh
coiner.BuildFluidMeshFromVelocityField()

# Build the pointLocators
coiner.BuildPointLocators()

# Run the simulation
coiner.Run(NperLine, Nlines, meanMode=meanMode, coinDirection=coinDirection, detailed_outcome=False, compute_ground_estimate=False,
           use_plane_fit=use_plane_fit, useConstraint=False)

if savefig:
    coiner.fig.savefig(savefile, dpi=200)
    print(f'Figure saved:\n{savefile}')

if save_velfield:
    velfield_ids = sum(coiner.fluidmeshpoints_to_fit.values(), [])
    velfield_to_save = coiner.velfield[velfield_ids, :]
    np.save(savefilepath, velfield_to_save)
    print(f'Saved velocity field inside coin stack as:\n{savefilepath}')
    
#%%
# Plot the results
# Plot the coins
plotted1 = False
plotted2 = False

fontsize = 28
titlesize = 24
labelsize = 22

MULTIPLIER = True
multilines = len(coiner.coin_hits) > 1

if multilines:
    fig, ax = plt.subplots(figsize=(16, 7))
else:
    fig, ax = plt.subplots(figsize=(8,10))

for j, coin_hits_in_list in enumerate(coiner.coin_hits):
    for i, coin_hits_in_bin in enumerate(coin_hits_in_list):
        idx = i + j * NperLine
        
        coin_center = coiner.outputmesh_vtk.GetPoint(idx) 
        vel_avg = coiner.u[idx]
        
        tracer_vel = coin_hits_in_bin[:,3]
        coin_centersZ = np.ones_like(tracer_vel) * coin_center[2]
        coin_centersX = np.ones_like(tracer_vel) * coin_center[0]
        
        if not plotted1:
            if multilines:
                ax.scatter(tracer_vel*MULTIPLIER + coin_centersX, coin_centersZ, marker='.', s=10, c='tab:blue', label=r'$u_{tracer}$', clip_on=False)
            else:
                ax.scatter(tracer_vel, coin_centersZ, marker='.', s=10, c='tab:blue', label=r'$u_{tracer}$')
            plotted1=True
        else:
            if multilines:
                ax.scatter(tracer_vel*MULTIPLIER + coin_centersX, coin_centersZ, marker='.', s=10, c='tab:blue', clip_on=False)
            else:
                ax.scatter(tracer_vel, coin_centersZ, marker='.', s=10, c='tab:blue')
    
    if not plotted2:
        if multilines:
            ax.plot(np.ones_like(zrange)*coin_center[0], zrange, '--', lw=3, color='k', clip_on=False)
            ax.plot(coiner.u[j*NperLine:(j+1)*NperLine]*MULTIPLIER + coin_center[0], zrange, '-x', ms=10, color='tab:red', label=r'$u_{mean}$', clip_on=False)
        else:
            ax.plot(coiner.u[j*NperLine:(j+1)*NperLine], zrange, '-x', ms=10, color='tab:red', label=r'$u_{mean}$')
        plotted2 = True
    else:
        if multilines:
            ax.plot(np.ones_like(zrange)*coin_center[0], zrange, '--', lw=3, color='k', clip_on=False)
            ax.plot(coiner.u[j*NperLine:(j+1)*NperLine]*MULTIPLIER + coin_center[0], zrange, '-x', ms=10, color='tab:red', clip_on=False)
        else:
            ax.plot(coiner.u[j*NperLine:(j+1)*NperLine], zrange, '-x', ms=10, color='tab:red')
        
ax.set_ylabel('z [mm]', fontsize=fontsize)
if multilines:
    ax.set_xlabel(r'x + $u \cdot 0.5 \cdot 10^{-3}$ [mm]', fontsize=fontsize)
    ax.set_title(f'Evolution of boundary layer velocity profile at y = {y}', fontsize=titlesize)
    ax.set_xlim(xmin-5, xmax+15)
    ax.legend(fontsize=fontsize, loc='upper left')
else:
    ax.set_xlabel('u [m/s]', fontsize=fontsize)
    ax.set_title(f'Coin-stacked velocity at (x, y) = ({xmin}, {y})', fontsize=titlesize)
    if xmin <= -150:
        ax.set_xlim(-2, 10)
    else:
        ax.set_xlim(-8, 10)
    ax.legend(fontsize=fontsize)
ax.set_ylim(zmin, zmax)


# ax.legend(fontsize=16, loc='upper right')#, bbox_to_anchor=(0.5,1.0))

# Add minor ticks
ax.minorticks_on()
# Resize ticks
ax.tick_params(axis='both', which='major', labelsize=labelsize)
ax.grid(visible=True, which='major', color='black',
          linewidth=0.7)
ax.grid(visible=True, which='minor', color='grey',
          linewidth=0.5, linestyle='-.')

if multilines:
    filename = f'CoinStackLineFrom-x={xmin}-to-x={xmax}-andAt-y={y}.png'
else:
    filename = f'CoinStackFitAt-x={xmin}-and-y={y}.png'
filepath = filedir / filename
fig.tight_layout()
# fig.savefig(filepath, dpi=200)