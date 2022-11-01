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

def perform_scm(params, sim_index, u_G_param):
    # Define file name for initial conditions
    params.init_path = output.init_directory + params.init_cond_path + 'U' + str(u_G_param) + '_'

    # Update parameter
    params.u_G = u_G_param
    params.sim_index = sim_index

    # Save parameter specifications in file for later reference
    save_solution.save_parameters_in_file(params, output, file_spec=str(u_G_param))

    # Solve model
    solve_turb_model(fparams, params, output)

    # Plot solution
    #plot_solution.make_3d_plot(output, params, fparams, file_spec=str(u_G_param))


# Define list of parameters for which the model shall be run (atm only u_G)
param_list = np.arange(1.0, 20.5, 0.5)

# Run model in parallel
num_proc = 50#multiprocessing.cpu_count() - 1
with multiprocessing.Pool(processes=num_proc) as pool:
    pool.map(partial(perform_scm, params, 1), param_list)






