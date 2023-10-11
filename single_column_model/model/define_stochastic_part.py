import numpy as np


def make_stochastic_grid(params, z):
    H_s_det_idx, H_s, N_s, delta_s = calculate_stochastic_grid_parameter(params, z)
    stoch_grid = define_grid_for_stochastic_part(params.z0, H_s, N_s)

    return stoch_grid, H_s_det_idx


def calculate_stochastic_grid_parameter(model_params, det_grid):
    # Step 1: Define distance between grid points
    # The distance between grid points on the stochastic grid can be at most the smallest grid point distance of the
    # deterministic grid
    delta_s = det_grid[1] - det_grid[0]

    # Step 2: Height of stochastic grid
    # Calculate height of stochastic grid
    H_s_total = model_params.H_sl * model_params.stoch_domain_ext
    # Find corresponding grid point of the deterministic grid
    H_s_total_det_idx = np.abs(det_grid - H_s_total).argmin()
    H_s_total_det = det_grid[H_s_total_det_idx]
    # The stochastic grid starts at s=0 not z0
    H_s_total = H_s_total_det

    # Step 3: Number of grid points
    N_s = int(np.ceil(H_s_total / delta_s))

    # Step 4: Recalculate distance between grid points to account for rounding in step 3
    delta_s = H_s_total / N_s

    return H_s_total_det_idx, H_s_total, N_s, delta_s


def define_grid_for_stochastic_part(lowest_grid_point, height, num_grid_points):
    return np.linspace(lowest_grid_point, height, num_grid_points)


def get_stoch_stab_function_parameter(richardson_num):
    return (define_stoch_stab_function_param_Lambda(richardson_num),
            define_stoch_stab_function_param_Upsilon(richardson_num),
            define_stoch_stab_function_param_Sigma(richardson_num))


def define_stoch_stab_function_param_Lambda(Ri):
    lambda_1 = 9.3212
    lambda_2 = 0.9088
    lambda_3 = 0.0738
    lambda_4 = 8.3220
    return lambda_1 * np.tanh(lambda_2 * np.log(Ri) - lambda_3) + lambda_4


def define_stoch_stab_function_param_Upsilon(Ri):
    upsilon_1 = 0.4294
    upsilon_2 = 0.1749
    return 10 ** (upsilon_1 * np.log(Ri) + upsilon_2)


def define_stoch_stab_function_param_Sigma(Ri, sigma_s=0.0):
    sigma_1 = 0.8069
    sigma_2 = 0.6044
    sigma_3 = 0.8368
    return 10 ** (sigma_1 * np.tanh(sigma_2 * np.log(Ri) - sigma_3) + sigma_s)
