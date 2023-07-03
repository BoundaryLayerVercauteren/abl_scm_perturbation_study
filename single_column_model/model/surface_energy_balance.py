# standard imports
import matplotlib.pyplot as plt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py

# project related imports
from single_column_model.model import define_PDE_model as fut
from single_column_model.utils import save_solution as ss
from single_column_model.model import solve_PDE_model as ut


def RHS_surf_balance_euler(x0, fenics_params, params, Ts, u_now, v_now, Kh_now, perturbation_val):
    # unroll params
    C_g = params.C_g
    k_m = params.k_m
    dt = params.dt
    th_m = params.theta_m

    # Calculate turbulent heat flux at the ground
    H_0 = get_heat_flux_at_ground(fenics_params, params, Ts, u_now, v_now, Kh_now)

    # Either use given net radiation or calculate it
    if params.R_n != 999:
        R_n = params.R_n
    else:
        R_n = calcuate_net_radiation(params, x0)

    # Add perturbation to net radiation if specified in parameter class
    if params.perturbation_param == 'net_rad':
        R_n = R_n + perturbation_val

    return x0 + (1.0 / C_g * (R_n - H_0) - k_m * (x0 - th_m)) * dt


def calcuate_net_radiation(params, theta_g_n):
    """
    Calculate incoming long-wave radiation (according to Maroneze 2019)

    Args:
        params (class): Class which contains parameters.
        theta_g_n (float): Surface temperature at time step t_n.

    Returns:
        (float): net radiation
    """
    incoming_longwave_radiation = params.sigma * (
                params.Q_c + 0.67 * (1 - params.Q_c) * (1670 * params.Q_a) ** 0.08) * params.theta_A ** 4
    return incoming_longwave_radiation - params.sigma * theta_g_n ** 4


def get_heat_flux_at_ground(fenics_params, params, Ts, u_now, v_now, Kh_now):
    # unroll params
    rho = params.rho
    C_p = params.C_p
    kappa = params.kappa
    z0 = params.z0
    z0h = params.z0h

    # calc some variables for the surface balance equation
    u_star, theta_star, Pr_turb = calc_ground_varibles(fenics_params, params, Ts, u_now, v_now, Kh_now)

    return -rho * C_p * theta_star * u_star * Pr_turb / kappa * np.log(z0 / z0h)


def get_temp_at_BL_top(ks, Ts):
    ks_array = ks.vector().get_local()
    m_w1 = np.max(ks_array)
    lim_m_w1 = 0.05 * m_w1  # 5 % of the maximum tke
    ind_h = ut.find_nearest(ks_array, lim_m_w1)

    return Ts.vector().get_local()[ind_h]


def calc_ground_varibles(fenics_params, params, Ts, u_now, v_now, Kh_now):
    u = u_now[1]
    v = v_now[1]
    Kh = Kh_now[1]

    # the value of these variable is at the ground level (z = 0)
    u_star = u_star_at_the_ground(fenics_params.z, u, v, params)
    theta_star = theta_star_at_the_ground(project(Ts.dx(0), fenics_params.Q), Kh, u_star)
    Pr_turb = Pr_turb_at_the_ground(fenics_params, params)

    return u_star, theta_star, Pr_turb


def update_tke_at_the_surface(fenics_params, params, u_now, v_now):
    u = u_now[1]
    v = v_now[1]

    fenics_params.k_D_low.value = np.max([k_at_the_ground(params, fenics_params.z, u, v), params.min_tke])


def u_star_at_the_ground(z, u_z1, v_z1, params):
    V_z1 = np.sqrt(u_z1 ** 2 + v_z1 ** 2)
    return params.kappa / np.log(z[1][0] / z[0][0]) * V_z1


def theta_star_at_the_ground(grad_T, K_h, u_star):
    # cast the gradient to numpy 
    get_grad_T = np.flipud(grad_T.vector().get_local())

    # get the flux for the full domain
    temp_turm_flux = K_h * get_grad_T

    # use flux at z1 (the lowest is "z0") to calc theta_star
    # the input "u_star" is the estimate from z1 height value
    theta_star = temp_turm_flux[1] / u_star

    return theta_star


def Pr_turb_at_the_ground(fenics_params, params):
    # input is a fenics variable
    full_Pr = project(fut.f_m(fenics_params, params) / fut.f_h(fenics_params, params), fenics_params.Q)

    # cast to numpy array
    get_full_Pr = np.flipud(full_Pr.vector().get_local())
    return get_full_Pr[0]


def k_at_the_ground(params, z, u_z1, v_z1, _f_m=0.087):
    return u_star_at_the_ground(z, u_z1, v_z1, params) ** 2 / np.sqrt(_f_m)
