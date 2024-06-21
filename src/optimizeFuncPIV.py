# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:02:49 2024

@author: ErikD
"""
import numpy as np
import scipy as sp

# File for optimization functions
def LSR_with_constraints(A, b, C, d):
    # Solves the least square regression ||Ax - b||^2 subject to constraints
    # Cx = d
    N_constraints = np.ma.size(C, axis=0)
    N_variables = np.ma.size(C, axis=1)
    
    LHS_row1 = np.hstack( (2*np.dot(A.T, A), C.T))
    LHS_row2 = np.hstack( (C, np.zeros((N_constraints, N_constraints))) )
    LHS = np.vstack((LHS_row1, LHS_row2))
    
    RHS = np.vstack((2*np.dot(A.T, b), d))
    
    sol = np.linalg.solve(LHS, RHS)
    
    out = {}
    out['x'] = sol[:N_variables]
    out['lagrangian'] = sol[N_variables:]
    
    return out

def optimizeLSR_wCons(x, vel, x_proj, panel=None, use_all_constraints=False):
    # Construct the least squares matrix, made up of the three sub X-matrices
    X_LSsub = X_for_tri_polynomial(x)
    
    Nvar = np.ma.size(X_LSsub, axis=1)
    
    # Define the fill (zero) matrix
    fill_matrix = np.zeros_like(X_LSsub)
    X_LSrow1 = np.hstack((X_LSsub,     fill_matrix, fill_matrix))
    X_LSrow2 = np.hstack((fill_matrix, X_LSsub,     fill_matrix))
    X_LSrow3 = np.hstack((fill_matrix, fill_matrix, X_LSsub))
    X_LS = np.vstack((X_LSrow1, X_LSrow2, X_LSrow3))
    
    # Define the Y matrix
    Y_LS = vel.reshape((-1,1), order='F')
    
    ## Set up the linear constraints
    # Value constraint at wall
    phi_wall = np.array([[basis_functions(i, x_proj) for i in range(Nvar)]])
    fill_row = np.zeros_like(phi_wall)
    
    # Constrain the u-velocity at the wall
    C1 = np.c_[phi_wall, fill_row, fill_row]
    # Constrain the v-velocity at the wall
    C2 = np.c_[fill_row, phi_wall, fill_row]
    # Constrain the w-velocity at the wall
    C3 = np.c_[fill_row, fill_row, phi_wall]
    
    if panel is not None:
        # Constrain the flux through the wall panel
        S = panel['normal']
        
        # Constrain the flux tangential to the wall panel in two directions
        direction1 = panel['AB']
        direction2 = np.cross(direction1, S)
        
        # Initialise the arrays
        C4 = np.zeros((1, 3*Nvar))
        C5 = np.zeros((1, 3*Nvar))
        C6 = np.zeros((1, 3*Nvar))
        
        for i in range(np.ma.size(C4, axis=1)):
            idx = i // Nvar
            phi_star = 0
            for wall_point in panel['vertices']:
                phi_star += basis_functions(i, wall_point)
            
            # Constrain the flux through the wall panel
            C4[0,i] = phi_star * S[idx]
            
            # Constrain the flux tangential to the wall panel in two directions
            intermediate_vector = phi_star * (1-S[idx])
            C5[0,i] = intermediate_vector * direction1[idx]
            C6[0,i] = intermediate_vector * direction2[idx]
    
    # Combine all constraints
    if use_all_constraints and (panel is not None):
        C = np.vstack((C1, C2, C3, C4, C5, C6))
    else:
        C = np.vstack((C1, C2, C3))
    
    
    # Define the equalities of the linear constraints
    if use_all_constraints:
        d = np.zeros((6,1))
    else:
        d = np.zeros((3,1))
    
    result = LSR_with_constraints(X_LS, Y_LS, C, d)
    
    return result

def create_panel_dictionary(vtk_cell, point):
    # Initialize the dictionary
    panel = {}
    
    # Define the corner points
    panel['vertices'] = np.array([vtk_cell.GetPoints().GetData().GetTuple(i) for i in range(3)]) - point.reshape((1,-1))

    # Define the corner centroid
    panel['centroid'] = np.array([0, 0, 0])
    vtk_cell.ComputeCentroid(vtk_cell.GetPoints(), [0, 1, 2], panel['centroid'])
    panel['centroid'] = panel['centroid'].reshape((1,-1)) - point.reshape((1,-1))
    
    # Define the normal
    panel['normal'] = np.array([0, 0, 0])
    vtk_cell.ComputeNormal(panel['vertices'][0,:],
                           panel['vertices'][1,:],
                           panel['vertices'][2,:],
                           panel['normal'])
    panel['normal'].reshape((1,-1))
    
    # Add the centroid to the points
    panel['vertices'] = np.vstack((panel['vertices'], panel['centroid']))
    
    # Compute the vector AB
    panel['AB'] = panel['vertices'][1,:].T - panel['vertices'][0,:].T
    
    return panel
    
                        
def optimize(velfield, sphere_center, x_proj, panels):
    # Given velfiedl (Mx6 array with x, y, z, u, v, w)
    # Given sphere center (1x3 array with x, y, z)
    # Given projected center (1x3 array with x, y, z)
    # Given panels ((Q,) list with entries (4, 3) of the three points with
    #   bounding vertices + the centroid)
    
    # Define the constraints
    # cons = # There are no constraints atm
    
    beta0 = np.zeros((30,)) # initial guess for 30 parameters beta
    optimal_velocity = sp.optimize.minimize(fun, beta0,
                                            args=(velfield[:10,:3],
                                                  velfield[:10,3:6],
                                                  panels,
                                                  x_proj),
                                            method='BFGS',
                                            options={'disp': True})
                                            # constraints=cons)
    
    return optimal_velocity

# Function which will be optimized
def fun(beta, x, vel, panels, x_proj):
    # beta is (N,) array with N number of variables
    # x is (M, 3) array with coordinates of M measurement points
    # vel is (M,3) array with velocities at M measurement points
    # panels is (Q,) list with entries (4, 3) of the three points with bounding
    #   vertices + the centroid
    # x_proj is (1, 3) array with the position of the projected point
    
    # Construct the least squares matrix, made up of the three sub X-matrices
    X_LSsub = X_for_tri_polynomial(x)
    
    # Define the fill (zero) matrix
    fill_matrix = np.zeros_like(X_LSsub)
    X_LSrow1 = np.hstack((X_LSsub,     fill_matrix, fill_matrix))
    X_LSrow2 = np.hstack((fill_matrix, X_LSsub,     fill_matrix))
    X_LSrow3 = np.hstack((fill_matrix, fill_matrix, X_LSsub))
    X_LS = np.vstack((X_LSrow1, X_LSrow2, X_LSrow3))
    
    # Define the Y matrix
    Y_LS = vel.reshape((-1,1), order='F')
    
    # Define the cost function (minimizing the quadratic of 1) the LSR, 2) the velocity projected at the wall, 3) the fluxes and 4) the tangential velocities
    # cost = np.sum((np.dot(X_LS, beta) - Y_LS)**2, axis=None)
    cost = np.sum((np.dot(np.dot(X_LS.T, X_LS), beta) - np.dot(X_LS.T, Y_LS))**2, axis=None)
    
    # Compute the cost related to the velocity at the projected point
    cost_vel_projected = np.linalg.norm(vel3D_func(beta, x_proj))
    
    # Add it to the total cost
    cost += cost_vel_projected
    
    # Compute the cost related to each panel
    for Q in panels:
        # Compute the cost related to the flux at panel Q
        cost_vel_flux = vel3D_func_flux(beta, Q)
        # Compute the cost related to the tangential flow at panel Q
        cost_vel_tangflow = vel3D_func_tangflow(beta, Q)
        
        # Add each to the total cost
        cost += cost_vel_flux**2
        cost += cost_vel_tangflow[0]**2
        cost += cost_vel_tangflow[1]**2
        
    return cost

def X_for_tri_polynomial(points, order=2):
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

def vel3D_derivative_func(beta, x, derivative):
    # For 2nd order function
    dvel = np.zeros((3,1))
    
    if derivative == 'x':
        basis_functions_der = basis_function_derx
    elif derivative == 'y':
        basis_functions_der = basis_function_dery
    elif derivative == 'z':
        basis_functions_der = basis_function_derz
    
    for i, beta_i in enumerate(beta):
        idx = i // 10
        
        dvel[idx] += beta_i * basis_functions_der(i, x)
    
    return dvel

def vel3D_func_fast(beta, x):
    # input x = (N,3)
    #       beta = (N,1)
    xmat = np.c_[np.ones_like(x[:,0]),  # 1 * a0
                 x[:,0],                # x * a1
                 x[:,1],                # y * a2 
                 x[:,2],                # z * a3
                 x[:,0]**2,             # x**2 * a4
                 x[:,1]**2,             # y**2 * a5
                 x[:,2]**2,             # z**2 * a6
                 x[:,-1]*x[:,0],        # z*x * a7
                 x[:,0]*x[:,1],         # x*y * a8
                 x[:,1]*x[:,2]]         # y*z * a9
    # if len(beta) > 10:
    #     xmat = np.stack((xmat, xmat, xmat), axis=2)
    return np.matmul(xmat, beta)

def vel3D_func(beta, x):
    # For 2nd order function
    vel = np.zeros((3,1))
    
    for i, beta_i in enumerate(beta):
        idx = i // 10
        vel[idx] += beta_i * basis_functions(i, x)
    
    return vel

def basis_function_derx(idx, x):
    #a0
    if idx % 10 == 0:
        phi = 0
    #a1
    elif idx % 10 == 1:
        phi = 1
    #a2
    elif (idx - 2) % 10 == 0:
        phi = 0
    #a3
    elif (idx - 3) % 10 == 0:
        phi = 0
    #a4
    elif (idx - 4) % 10 == 0:
        phi = 2 * x[0]
    #a5
    elif (idx - 5) % 10 == 0:
        phi = 0
    #a6
    elif (idx - 6) % 10 == 0:
        phi = 0
    #a7
    elif (idx - 7) % 10 == 0:
        phi = x[-1]
    #a8
    elif (idx - 8) % 10 == 0:
        phi = x[1]
    #a9
    elif (idx - 9) % 10 == 0:
        phi = 0
    
    return phi

def basis_function_dery(idx, x):
    #a0
    if idx % 10 == 0:
        phi = 0
    #a1
    elif idx % 10 == 1:
        phi = 0
    #a2
    elif (idx - 2) % 10 == 0:
        phi = 1
    #a3
    elif (idx - 3) % 10 == 0:
        phi = 0
    #a4
    elif (idx - 4) % 10 == 0:
        phi = 0
    #a5
    elif (idx - 5) % 10 == 0:
        phi = 2 * x[1]
    #a6
    elif (idx - 6) % 10 == 0:
        phi = 0
    #a7
    elif (idx - 7) % 10 == 0:
        phi = 0
    #a8
    elif (idx - 8) % 10 == 0:
        phi = x[0]
    #a9
    elif (idx - 9) % 10 == 0:
        phi = x[1]
    
    return phi

def basis_function_derz(idx, x):
    #a0
    if idx % 10 == 0:
        phi = 0
    #a1
    elif idx % 10 == 1:
        phi = 0
    #a2
    elif (idx - 2) % 10 == 0:
        phi = 0
    #a3
    elif (idx - 3) % 10 == 0:
        phi = 1
    #a4
    elif (idx - 4) % 10 == 0:
        phi = 0
    #a5
    elif (idx - 5) % 10 == 0:
        phi = 0
    #a6
    elif (idx - 6) % 10 == 0:
        phi = 2 * x[2]
    #a7
    elif (idx - 7) % 10 == 0:
        phi = x[0]
    #a8
    elif (idx - 8) % 10 == 0:
        phi = 0
    #a9
    elif (idx - 9) % 10 == 0:
        phi = x[1]
    
    return phi


def basis_functions(idx, x):
    #a0
    if idx % 10 == 0:
        phi = 1
    #a1
    elif idx % 10 == 1:
        phi = x[0]
    #a2
    elif (idx - 2) % 10 == 0:
        phi = x[1]
    #a3
    elif (idx - 3) % 10 == 0:
        phi = x[2]
    #a4
    elif (idx - 4) % 10 == 0:
        phi = x[0]**2
    #a5
    elif (idx - 5) % 10 == 0:
        phi = x[1]**2
    #a6
    elif (idx - 6) % 10 == 0:
        phi = x[2]**2
    #a7
    elif (idx - 7) % 10 == 0:
        phi = x[-1]*x[0]
    #a8
    elif (idx - 8) % 10 == 0:
        phi = x[0]*x[1]
    #a9
    elif (idx - 9) % 10 == 0:
        phi = x[1]*x[2]
    
    return phi
 

def vel3D_func_flux(beta, panel):
    # Velocity to compute the flux through a panel Q, given the coefficients beta
    return 0.0

def vel3D_func_tangflow(beta, panel):
    # Velocity to compute the flow tangential to a panel Q, given the coefficients beta
    return (0.0, 0.0)