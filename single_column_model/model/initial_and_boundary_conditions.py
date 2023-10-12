# coding=utf-8
# !/usr/bin/env python

import fenics as fe
import numpy as np

# project related imports
from single_column_model.utils import transform_values as tv


def load_initial_conditions_from_files(file_path):
    u = np.load(file_path + "_u.npy")
    v = np.load(file_path + "_v.npy")
    theta = np.load(file_path + "_theta.npy")
    k = np.load(file_path + "_TKE.npy")

    return u, v, theta, k


def calculate_initial_conditions(mesh_coordinates, roughness_length, geo_wind_vert, kappa, domain_height,
                                 horiz_resolution, theta_reference, gamma):
    u = initial_u_0(mesh_coordinates, geo_wind_vert, roughness_length, kappa)
    v = 0 * np.ones(horiz_resolution)
    theta = initial_theta_0(mesh_coordinates, theta_reference, gamma, 200)
    k = initial_k_0(mesh_coordinates, geo_wind_vert, roughness_length, domain_height) + 0.01

    return u, v, theta, k


def transfer_all_variables_to_fenics_functions(var1, var2, var3, var4, function_space_projection):
    var1 = tv.convert_numpy_array_to_fenics_function(var1, function_space_projection)
    var2 = tv.convert_numpy_array_to_fenics_function(var2, function_space_projection)
    var3 = tv.convert_numpy_array_to_fenics_function(var3, function_space_projection)
    var4 = tv.convert_numpy_array_to_fenics_function(var4, function_space_projection)

    return var1, var2, var3, var4


def define_initial_conditions(Q, mesh, params):
    """Combine functions to either load initial conditions from files or calculate them ad hoc."""
    if params.load_ini_cond:
        u_t0, v_t0, theta_t0, k_t0 = load_initial_conditions_from_files(params.init_path)

    else:
        u_t0, v_t0, theta_t0, k_t0 = calculate_initial_conditions(mesh.coordinates(), params.z0, params.u_G,
                                                                  params.kappa, params.H, params.Nz, params.theta_ref,
                                                                  params.gamma)

    return transfer_all_variables_to_fenics_functions(u_t0, v_t0, theta_t0, k_t0, Q)


def define_boundary_conditions(fenics_params, params):
    z0 = params.z0  # roughness length in meter
    H = params.H  # domain height in meters
    u_G = params.u_G  # u geostrophic wind
    initCondStr = params.init_path  # name of the file

    load_ini_cond = params.load_ini_cond  # bool type; load existing initial condition

    V = fenics_params.W  # fenics variable; the vector function space

    ground = 'near(x[0],' + str(z0) + ',1E-6)'
    top = 'near(x[0],' + str(H) + ',1E-6)'

    if load_ini_cond:
        u_ini = np.load(initCondStr + '_u.npy')
        v_ini = np.load(initCondStr + '_v.npy')
        theta_ini = np.load(initCondStr + '_theta.npy')
        k_ini = np.load(initCondStr + '_TKE.npy')

        u_low = fe.Expression('value', degree=0, value=u_ini[0])
        u_top = fe.Expression('value', degree=0, value=u_ini[-1])

        v_low = fe.Expression('value', degree=0, value=v_ini[0])

        theta_low = fe.Expression('value', degree=0, value=theta_ini[0])

        k_low = fe.Expression('value', degree=0, value=k_ini[0])
        k_top = fe.Expression('value', degree=0, value=k_ini[-1])

        # velocity u component
        bc_u_ground = fe.DirichletBC(V.sub(0), u_low, ground)
        bcu_top = fe.DirichletBC(V.sub(0), u_top, top)

        # velocity v component
        bc_v_ground = fe.DirichletBC(V.sub(1), v_low, ground)
        bcv_top = fe.DirichletBC(V.sub(1), 0.0, top)

        # Temperature
        bc_theta_ground = fe.DirichletBC(V.sub(2), theta_low, ground)

        # TKE
        bck_ground = fe.DirichletBC(V.sub(3), k_low, ground)
        bck_top = fe.DirichletBC(V.sub(3), k_top, top)

        bc = [bc_u_ground, bc_v_ground, bc_theta_ground, bck_ground, bcv_top]

        Tg_n = theta_ini[0]

    else:
        # Define velocity u component
        bc_u_ground = fe.DirichletBC(V.sub(0), 0.0, ground)

        # Define velocity v component
        bc_v_ground = fe.DirichletBC(V.sub(1), 0.0, ground)
        bc_v_top = fe.DirichletBC(V.sub(1), 0.0, top)

        # Define potential temperature
        theta_low = fe.Expression('value', degree=0, value=params.theta_ref)
        bc_theta_ground = fe.DirichletBC(V.sub(2), theta_low, ground)

        # Define TKE
        k_low = fe.Expression('value', degree=0, value=initial_k_0(z0, u_G, z0, 200))
        bc_k_ground = fe.DirichletBC(V.sub(3), k_low, ground)

        # Combine boundary conditions
        bc = [bc_u_ground, bc_v_ground, bc_theta_ground, bc_k_ground, bc_v_top]

        #
        Tg_n = params.theta_ref

    # writing out the fenics parameters
    fenics_params.bc = bc  # list of boundary conditions. Will be used in the FEM formulation
    fenics_params.theta_D_low = theta_low  # Temperature. Fenics expression is used to control the value within the main loop solution
    fenics_params.k_D_low = k_low  # TKE. Fenics expression is used to control the value within the main loop solution

    fenics_params.U_g = fe.Expression('value', degree=0,
                                      value=params.u_G)  # Geostrophic wind; added here to control in the main loop
    fenics_params.V_g = fe.Constant(params.v_G)  # Geostrophic wind; added here to control in the main loop

    # writing out normal parameters
    params.Tg_n = Tg_n  # The value of the Temperature at the ground.

    return fenics_params, params


def initial_u_0(z, u_G, z0, kappa):
    """Calculate  initial profile for the wind velocity u.

        Args:
            z (fenics function): Mesh coordinates.
            u_G (float): Zonal geostrophic wind speed.
            z0 (float): Surface roughness length.
            params (class): Parameters from dataclass. For more details see parameters.py

        Returns:
            (list?): initial profile for u
    """
    # Set tuning parameter
    c_f = 4 * 10 ** (-3)
    # Calculate initial friction velocity
    u_star_ini = np.sqrt(0.5 * c_f * u_G ** 2)
    return u_star_ini / kappa * np.log(z / z0)


def initial_theta_0(z, theta_initial, gamma, cut_height):
    """Calculate  initial profile for the temperature theta.

    Args:
        z (fenics function): Mesh coordinates.
        theta_initial (float): Initial potential temperature.
        cut_height (float): Height at which the temperature depends on the laps rate and the height of the grid point.

    Returns:
        (list?): initial profile for theta
    """
    # Set initial theta above reference height
    theta_0 = (z - cut_height) * gamma + theta_initial
    # Find index of value where z - cut_height = 0
    index_cut_height = int(np.abs(z - cut_height).argmin())
    # Set initial theta below reference height
    theta_0[0:index_cut_height + 1] = theta_initial
    return theta_0


def initial_k_0(z, u_G, z0, H, k_at_H=0.0):
    """Calculate  initial profile for TKE. See Parente, A., C. Gorlé, J. van Beeck, and C. Benocci, 2011: A
    Comprehensive Modelling Approach for the Neutral Atmospheric Boundary Layer: Consistent Inflow Conditions, Wall
    Function and Turbulence Model. Boundary-Layer Meteorol, 140, 411–428, https://doi.org/10.1007/s10546-011-9621-5.

    Args:
        params (class): Parameters from dataclass. For more details see parameters.py.
        z (fenics function): Mesh coordinates.
        u_G (float): Zonal geostrophic wind speed.
        z0 (float): Surface roughness length.
        H (float): Domain height.
        k_at_H (float): TKE for t=0 and z=H.

    Returns:
        (list?): initial profile for TKE
    """
    # Set tuning parameter
    c_f = 4 * 10 ** (-3)
    # Calculate initial friction velocity
    u_star_ini = np.sqrt(0.5 * c_f * u_G ** 2)
    # TKE for t=0 and z=z0
    k_at_z0 = u_star_ini ** 2 / np.sqrt(0.087)
    # Solve system of equations
    func = lambda z, a, b: a * np.log(z + z0) + b
    a = (k_at_H - k_at_z0) / (np.log(H) - np.log(z0))
    b = k_at_z0 - a * np.log(z0)
    return func(z, a, b)
