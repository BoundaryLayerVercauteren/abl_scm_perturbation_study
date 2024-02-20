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


def calculate_initial_conditions(
    mesh_coordinates,
    roughness_length,
    geo_wind_vert,
    kappa,
    domain_height,
    horiz_resolution,
    theta_reference,
    gamma,
    initial_cond_perturbation,
):
    u = initial_u_0(
        mesh_coordinates,
        geo_wind_vert,
        roughness_length,
        kappa,
        initial_cond_perturbation,
    )
    v = 0 * np.ones(horiz_resolution)
    theta = initial_theta_0(mesh_coordinates, theta_reference, gamma, 200)
    k = (
        initial_k_0(mesh_coordinates, geo_wind_vert, roughness_length, domain_height)
        + 0.01
    )

    return u, v, theta, k


def transfer_all_variables_to_fenics_functions(
    var1, var2, var3, var4, function_space_projection
):
    var1 = tv.convert_numpy_array_to_fenics_function(var1, function_space_projection)
    var2 = tv.convert_numpy_array_to_fenics_function(var2, function_space_projection)
    var3 = tv.convert_numpy_array_to_fenics_function(var3, function_space_projection)
    var4 = tv.convert_numpy_array_to_fenics_function(var4, function_space_projection)

    return var1, var2, var3, var4


def define_initial_conditions(Q, mesh, params):
    """Combine functions to either load initial conditions from files or calculate them ad hoc."""
    if params.load_ini_cond:
        u_t0, v_t0, theta_t0, k_t0 = load_initial_conditions_from_files(
            params.init_path
        )

    else:
        u_t0, v_t0, theta_t0, k_t0 = calculate_initial_conditions(
            mesh.coordinates(),
            params.z0,
            params.u_G,
            params.kappa,
            params.H,
            params.Nz,
            params.theta_ref,
            params.gamma,
            params.initial_cond_perturbation,
        )

    return transfer_all_variables_to_fenics_functions(u_t0, v_t0, theta_t0, k_t0, Q)


def load_boundary_conditions_from_files(initial_cond_path):
    u_t0, v_t0, theta_t0, k_t0 = load_initial_conditions_from_files(initial_cond_path)

    u_z0 = fe.Expression("value", degree=0, value=u_t0[0])
    v_z0 = fe.Expression("value", degree=0, value=v_t0[0])
    theta_z0 = fe.Expression("value", degree=0, value=theta_t0[0])
    k_z0 = fe.Expression("value", degree=0, value=k_t0[0])

    theta_g_0 = theta_t0[0]

    return u_z0, v_z0, theta_z0, k_z0, theta_g_0


def turn_into_fenics_boundary_conditions(
    u_z0, v_z0, theta_z0, k_z0, vector_space, ground_def, top_def
):
    bc_u_ground = fe.DirichletBC(vector_space.sub(0), u_z0, ground_def)
    bc_v_ground = fe.DirichletBC(vector_space.sub(1), v_z0, ground_def)
    bc_theta_ground = fe.DirichletBC(vector_space.sub(2), theta_z0, ground_def)
    bc_k_ground = fe.DirichletBC(vector_space.sub(3), k_z0, ground_def)

    bc_v_top = fe.DirichletBC(vector_space.sub(1), 0.0, top_def)

    return [bc_u_ground, bc_v_ground, bc_theta_ground, bc_k_ground, bc_v_top]


def define_boundary_conditions(fenics_params, params):
    """Combine functions to either load boundary conditions from files or calculate them ad hoc."""
    ground = f"near(x[0],{params.z0},1E-6)"
    top = f"near(x[0],{params.H},1E-6)"

    if params.load_ini_cond:
        u_ground, v_ground, theta_ground, k_ground, theta_g_0 = (
            load_boundary_conditions_from_files(params.init_path)
        )

        boundary_cond = turn_into_fenics_boundary_conditions(
            u_ground, v_ground, theta_ground, k_ground, fenics_params.W, ground, top
        )

    else:
        theta_ground = fe.Expression("value", degree=0, value=params.theta_ref)
        k_ground = fe.Expression(
            "value", degree=0, value=initial_k_0(params.z0, params.u_G, params.z0, 200)
        )

        boundary_cond = turn_into_fenics_boundary_conditions(
            0.0, 0.0, theta_ground, k_ground, fenics_params.W, ground, top
        )

        theta_g_0 = params.theta_ref

    fenics_params.bc = boundary_cond
    fenics_params.theta_D_low = theta_ground
    fenics_params.k_D_low = k_ground

    fenics_params.U_g = fe.Expression("value", degree=0, value=params.u_G)
    fenics_params.V_g = fe.Constant(params.v_G)

    params.theta_g_n = theta_g_0

    # Initial condition for stochastic stability function
    fenics_params.f_ms = fe.Expression("value", value=fe.Constant(1.0), degree=0)

    return fenics_params, params


def initial_u_0(z, u_G, z0, kappa, initial_cond_perturbation=0):
    """Calculate  initial profile for the wind velocity u."""
    # Set tuning parameter
    c_f = 4 * 10 ** (-3)
    # Calculate initial friction velocity
    u_star_ini = np.sqrt(0.5 * c_f * (u_G + initial_cond_perturbation) ** 2)
    return u_star_ini / kappa * np.log(z / z0)


def initial_theta_0(z, theta_initial, laps_rate, cut_height):
    """Calculate  initial profile for the temperature theta."""
    # Set initial theta above reference height
    theta_0 = (z - cut_height) * laps_rate + theta_initial
    # Find index of value where z - cut_height = 0
    index_cut_height = int(np.abs(z - cut_height).argmin())
    # Set initial theta below reference height
    theta_0[0 : index_cut_height + 1] = theta_initial
    return theta_0


def initial_k_0(z, u_G, z0, H, k_at_H=0.0):
    """Calculate  initial profile for TKE. See Parente, A., C. Gorlé, J. van Beeck, and C. Benocci, 2011: A
    Comprehensive Modelling Approach for the Neutral Atmospheric Boundary Layer: Consistent Inflow Conditions, Wall
    Function and Turbulence Model. Boundary-Layer Meteorol, 140, 411–428, https://doi.org/10.1007/s10546-011-9621-5.
    """
    # Set tuning parameter
    c_f = 4 * 10 ** (-3)
    # Calculate initial friction velocity
    u_star_ini = np.sqrt(0.5 * c_f * u_G**2)
    # TKE for t=0 and z=z0
    k_at_z0 = u_star_ini**2 / np.sqrt(0.087)

    def tke_profile(z, a, b):
        return a * np.log(z + z0) + b

    a = (k_at_H - k_at_z0) / (np.log(H) - np.log(z0))
    b = k_at_z0 - a * np.log(z0)
    return tke_profile(z, a, b)


def update_tke_at_the_surface(kappa, z, u, v, min_tke):
    return np.max([calculate_tke_at_the_ground(kappa, z, u[1], v[1]), min_tke])


def calculate_tke_at_the_ground(kappa, z, u_z1, v_z1, c_f_m=0.087):
    return calculate_u_star_at_the_ground(kappa, z, u_z1, v_z1) ** 2 / np.sqrt(c_f_m)


def calculate_u_star_at_the_ground(kappa, z, u_z1, v_z1):
    return kappa / np.log(z[1][0] / z[0][0]) * np.sqrt(u_z1**2 + v_z1**2)
