import numpy as np


# coding=utf-8
# !/usr/bin/env python


def define_abraham_function(num_steps, T_end, z, t_k, r):
    """
    Create 2D space and time function which is a modified version of the one defined in

    'Abraham, C., A. M. Holdsworth, and A. H. Monahan, 2019: A prototype stochastic parameterization of regime behaviour
    in the stably stratified atmospheric boundary layer. Nonlinear Processes in Geophysics, 26, 401–427,
    https://doi.org/10.5194/npg-26-401-2019.'

    Args:
        num_steps (int): Number of steps in time.
        dt (int): Length of time steps.
        z (numpy array): Array with values for space axes.
        t_k (int): Point in time when pulse starts.
        r (int): Maximal strength of pulse.

    Returns:
        (numpy array): Values of modified Abraham function.
    """
    t = np.linspace(0, T_end, num_steps)

    s_k = np.zeros_like(t)
    h_k = np.zeros_like(t)
    sigma_k = np.zeros_like(t)

    SF_k = np.zeros([np.shape(z)[0], len(t)])

    # Set parameters for random process
    tau_e = 1200  # eddey overturning timescale
    tau_w = 60  # growth time
    t_wk = t_k + tau_w
    h_b = 75  # centre of turbulent pulse at t0
    h_e = 15  # centre of turbulent pulse at the end
    tau_h = 900  # vertical migration timescale of the centre
    sigma_w = 30  # width of turbulent pulse at t0
    sigma_e = 50  # width of turbulent pulse at the end
    tau_sigma = 900  # broadening timescale

    for t_idx, t_curr in enumerate(t):
        if t_k <= t_curr < t_wk:
            h_k[t_idx] = h_b
            sigma_k[t_idx] = (sigma_w + 1) / 2 * np.tanh(
                (t_curr - 0.5 * tau_w - t_wk) / (0.5 * tau_w) * np.arctanh((sigma_w - 1) / (sigma_w + 1))) + (
                                     sigma_w + 1) / 2
            s_k[t_idx] = (0.505 * r * np.tanh(
                (t_curr - 0.5 * tau_w - t_wk) / (0.5 * tau_w) * np.arctanh(99 / 101)) + 0.505 * r)

        elif t_curr >= t_wk:
            s_k[t_idx] = r * np.exp(-(t_curr - t_wk) / tau_e)
            h_k[t_idx] = -(h_b - h_e) * np.exp(-(t_curr - t_wk) / tau_h) + h_e
            sigma_k[t_idx] = (sigma_w - sigma_e) * np.exp(-(t_curr - t_wk) / tau_sigma) + sigma_e

        if t_curr > t_k:
            SF_k[:, t_idx] = (s_k[t_idx] * np.exp(-((-z - h_k[t_idx]) ** 2) / (2 * sigma_k[t_idx] ** 2))).flatten()

    return SF_k


def create_space_time_abraham_perturbation(num_steps, perturbation_start, T_end, z, pulse_strength):
    t_k = perturbation_start

    return pulse_strength, define_abraham_function(num_steps, T_end, z, t_k, pulse_strength)


def two_dim_gaussian_function(num_steps, T_end, z, start, amplitude, time_spread, height_spread, dt):
    t = np.linspace(0, T_end, num_steps)
    time_perturb_center = start*dt + time_spread / 2
    height_perturb_center = 20
    gaussian = amplitude * np.exp(
        -((t - time_perturb_center) ** 2 / (2 * time_spread ** 2) + (z - height_perturb_center) ** 2 / (
                2 * height_spread ** 2)))
    return amplitude, gaussian


def create_space_time_perturbation(params, fenics_params):
    """
    Create 2D perturbation (for the space and time dimension).

    Args:
        params (class): Parameter class for simulation specific parameters.
        fenics_params (class): Parameter class for fenics parameters (i.e. mesh, function space).

    Returns:
        (numpy array): A perturbation for all z and t.
    """
    if "mod_abraham" == params.perturbation_type:
        pulse_strength_val, perturbation_val = create_space_time_abraham_perturbation(
            params.num_steps,
            params.perturbation_start,
            params.T_end,
            fenics_params.z,
            params.perturbation_strength
        )
    elif "neg_mod_abraham" == params.perturbation_type:
        pulse_strength_val, perturbation_val = create_space_time_abraham_perturbation(
            params.num_steps,
            params.perturbation_start,
            params.T_end,
            fenics_params.z,
            params.perturbation_strength
        )
        perturbation_val = -1.0 * perturbation_val
        pulse_strength_val = -1.0 * pulse_strength_val
    elif "pos_gaussian" == params.perturbation_type:
        pulse_strength_val, perturbation_val = two_dim_gaussian_function(params.num_steps, params.T_end,
                                                                         fenics_params.z, params.perturbation_start,
                                                                         params.perturbation_strength,
                                                                         params.perturbation_time_spread,
                                                                         params.perturbation_height_spread,
                                                                         params.dt)
    elif "neg_gaussian" == params.perturbation_type:
        pulse_strength_val, perturbation_val = two_dim_gaussian_function(params.num_steps, params.T_end,
                                                                         fenics_params.z, params.perturbation_start,
                                                                         params.perturbation_strength,
                                                                         params.perturbation_time_spread,
                                                                         params.perturbation_height_spread,
                                                                         params.dt)
        perturbation_val = -1.0 * perturbation_val
        pulse_strength_val = -1.0 * pulse_strength_val
    else:
        raise ValueError(f'The specified perturbation type {params.perturbation_type} is not valid. Valid options are:'
                         f'(neg_)mod_abraham and (neg)pos_gaussian if {params.perturbation_param} is perturbed.')

    return pulse_strength_val, perturbation_val


def create_time_gauss_process_perturbation(num_steps, perturbation_length, perturb_start, perturbation_strength):
    """
    Create 1D (time) perturbation which is a Gauss process.

    Args:
        num_steps (int): Number of steps in time.
        dt (int): Length of time steps.
        z (numpy array): Array with values for space axes.

    Returns:
        (numpy array): Values of perturbation.
    """
    gauss_perturbation = np.abs(np.random.normal(0.0, perturbation_strength, perturbation_length))
    perturbation = np.concatenate((np.zeros(perturb_start - 1), gauss_perturbation)).reshape(1, num_steps)

    return perturbation_strength, perturbation


def create_time_perturbation(params):
    if "gauss_process" == params.perturbation_type:
        pulse_strength_val, perturbation_val = create_time_gauss_process_perturbation(
            params.num_steps,
            params.perturbation_length,
            params.perturbation_start,
            params.perturbation_strength,
        )

    return pulse_strength_val, perturbation_val


def create_perturbation(params, fenics_params, output):
    # Calculate perturbation
    if "pde" in params.perturbation_param:
        pulse_strength, perturbation = create_space_time_perturbation(
            params, fenics_params
        )
        output.r = pulse_strength
    elif params.perturbation_param == "net_rad":
        pulse_strength, perturbation = create_time_perturbation(params)
        output.r = pulse_strength
    else:
        perturbation = np.empty([1, params.num_steps])
        output.r = np.nan

    # Save perturbation in output class
    output.perturbation = perturbation

    return output
