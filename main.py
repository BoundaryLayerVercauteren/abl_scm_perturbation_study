# project related imports
from single_column_model.model.combine_solver_functions import solve_turb_model
from single_column_model.model import parameters as pc
from single_column_model.utils import save_solution, plot_solution

import multiprocessing
from functools import partial
import numpy as np

# Initialize parameters
params, fparams, output = pc.initialize_project_variables()

# Create directory for solutions and initial conditions (if required)
output.solution_directory, output.init_directory = save_solution.create_solution_directory(params)


def perform_scm(params, u_G_param, sim_index=1):
    # Round parameter
    u_G_param = np.around(u_G_param, 1)

    # Define file name for initial conditions
    params.init_path = output.init_directory + params.init_cond_path + 'Ug' + str(u_G_param)

    # Update parameter
    params.u_G = u_G_param
    params.sim_index = sim_index

    # Save parameter specifications in file for later reference
    if sim_index == 1:
        save_solution.save_parameters_in_file(params, output, file_spec=str(u_G_param))

    # Solve model
    solve_turb_model(fparams, params, output)

    # Plot solution
    #plot_solution.make_3d_plot(output, params, fparams, file_spec=str(u_G_param) + '_' + str(sim_index))


# Define list of parameters for which the model shall be run (atm only u_G)
if params.perturbation_param == 'pde_u' and params.perturbation_type == 'mod_abraham':
    param_list = [2.1]
elif params.perturbation_param == 'pde_u' and params.perturbation_type == 'neg_mod_abraham':
    param_list = [2.4]
elif params.perturbation_param == 'pde_theta' and params.perturbation_type == 'mod_abraham':
    param_list = [2.4]
elif params.perturbation_param == 'pde_theta' and params.perturbation_type == 'neg_mod_abraham':
    param_list = [2.1]
elif params.perturbation_param == 'none':
    param_list = np.arange(1.0, 7.0, 0.2)

# Run model in parallel
if params.num_simulation == 1:
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        pool.map(partial(perform_scm, params), param_list)
else:
    with multiprocessing.Pool(processes=params.num_proc) as pool:
        if len(param_list) > 1:
            for param_val in param_list:
                pool.map(partial(perform_scm, params, param_val), range(params.num_simulation))
        else:
            pool.map(partial(perform_scm, params, param_list[0]), range(params.num_simulation))

print('All simulations are done.')
