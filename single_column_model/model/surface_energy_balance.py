# coding=utf-8
#!/usr/bin/env python

# standard imports
import numpy as np

# project related imports
from single_column_model.utils import transform_values


def calculate_surface_temperature_euler_form(theta_g_n, fenics_params, params, theta_n, Kh_n, perturbation_val):

    # Calculate turbulent heat flux at the ground
    H_0 = calculate_heat_flux_at_ground(fenics_params, params, theta_n, Kh_n)

    # Add perturbation to net radiation if specified in parameter class
    if params.perturbation_param == "net_rad":
        R_n = params.R_n + perturbation_val
    else:
        R_n = params.R_n

    return theta_g_n + (1.0 / params.C_g * (R_n - H_0) - params.k_m * (theta_g_n - params.theta_m)) * params.dt


def calculate_heat_flux_at_ground(fenics_params, params, theta, K_h):
    temp_flux_surface = calculate_vertical_temperature_flux_at_surface(theta.dx(0), fenics_params.Q, K_h[1])

    return -params.rho * params.C_p * temp_flux_surface * params.Pr_t / params.kappa * np.log(params.z0 / params.z0h)


def calculate_vertical_temperature_flux_at_surface(grad_theta, function_space_projection, K_h):
    # Cast the gradient to numpy array
    grad_theta_values = transform_values.project_fenics_function_to_numpy_array(grad_theta, function_space_projection)

    # Calculate flux for the full domain
    vertical_temperature_flux = K_h * grad_theta_values

    return vertical_temperature_flux[0]
