import numpy as np


def two_dim_gaussian_function(
    num_steps, T_end, z, start, amplitude, time_spread, height_spread, dt
):
    t = np.linspace(0, T_end, num_steps)
    time_perturb_center = start * dt + time_spread / 2
    height_perturb_center = 20
    gaussian = amplitude * np.exp(
        -(
            (t - time_perturb_center) ** 2 / (2 * time_spread**2)
            + (z - height_perturb_center) ** 2 / (2 * height_spread**2)
        )
    )
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
    if "pos" == params.perturbation_type:
        pulse_strength_val, perturbation_val = two_dim_gaussian_function(
            params.num_steps,
            params.T_end,
            fenics_params.z,
            params.perturbation_start,
            params.perturbation_strength,
            params.perturbation_time_spread,
            params.perturbation_height_spread,
            params.dt,
        )
    elif "neg" == params.perturbation_type:
        pulse_strength_val, perturbation_val = two_dim_gaussian_function(
            params.num_steps,
            params.T_end,
            fenics_params.z,
            params.perturbation_start,
            params.perturbation_strength,
            params.perturbation_time_spread,
            params.perturbation_height_spread,
            params.dt,
        )
        perturbation_val = -1.0 * perturbation_val
        pulse_strength_val = -1.0 * pulse_strength_val
    else:
        raise ValueError(
            f"The specified perturbation type {params.perturbation_type} is not valid. Valid options are:"
            f"(neg)pos if {params.perturbation_param} is perturbed."
        )

    return pulse_strength_val, perturbation_val


def create_time_gauss_process_perturbation(
    num_steps, perturbation_length, perturb_start, perturbation_strength
):
    """
    Create 1D (time) perturbation which is a Gauss process.

    Args:
        num_steps (int): Number of steps in time.
        dt (int): Length of time steps.
        z (numpy array): Array with values for space axes.

    Returns:
        (numpy array): Values of perturbation.
    """
    gauss_perturbation = np.abs(
        np.random.normal(0.0, perturbation_strength, perturbation_length)
    )
    perturbation = np.concatenate(
        (np.zeros(perturb_start - 1), gauss_perturbation)
    ).reshape(1, num_steps)

    return perturbation_strength, perturbation


def create_time_perturbation(params):
    if "gauss_process" == params.perturbation_type:
        pulse_strength_val, perturbation_val = create_time_gauss_process_perturbation(
            params.num_steps,
            params.perturbation_time_spread,
            params.perturbation_start,
            params.perturbation_strength,
        )

    return pulse_strength_val, perturbation_val


def create_perturbation(params, fenics_params, output):
    # Calculate perturbation
    if (
        params.perturbation_param == "u"
        or params.perturbation_param == "theta"
        or params.perturbation_param == "u and v"
    ):
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
