import multiprocessing
import numpy as np
import sys
from functools import partial

if "pytest" in sys.modules:
    from single_column_model.tests import parameters
else:
    from single_column_model.model import parameters

# Project related imports
from single_column_model.model import \
    space_discretization  # define_stochastic_part,
from single_column_model.model import (define_initial_and_boundary_conditions,
                                       define_PDE_model, define_perturbation,
                                       define_parts_for_stoch_stability_function, solve_PDE_model)
from single_column_model.utils import save_solution


def make_setup_for_model_run(create_dir=True):
    # Initialize parameters
    params, fparams, output = parameters.initialize_project_variables()

    # To make naming of output files correct
    if params.perturbation_type == 'none' and params.perturbation_param != 'stab_func':
        params.perturbation_strength = 'nan'

    if create_dir:
        # Create directory for solutions and initial conditions (if required)
        output.solution_directory, output.init_directory = save_solution.create_solution_directory(params)

    return params, fparams, output


def combine_model_solver_functions(fenics_params, params, output):
    # Create deterministic mesh; options: 'power' , 'log', 'log_lin'
    mesh = space_discretization.create_grid(params, "power")

    # Create stochastic grid
    params.stoch_grid, params.Hs_det_idx, params.Hs, params.Ns_n, params.dz_s = define_parts_for_stoch_stability_function.make_stochastic_grid(
        params, mesh.coordinates())

    # Define variables to use the fenics library
    fenics_params = define_PDE_model.setup_fenics_variables(fenics_params, mesh)

    # Define boundary conditions
    fenics_params, params = define_initial_and_boundary_conditions.define_boundary_conditions(fenics_params, params)

    # Define initial profiles/ values
    u_n, v_n, T_n, k_n = define_initial_and_boundary_conditions.define_initial_conditions(fenics_params.Q, mesh, params)

    # Set up the weak formulation of the equations
    fenics_params.F = define_PDE_model.weak_formulation(fenics_params, params, u_n, v_n, T_n, k_n)

    # Set up the stochastic solver
    stoch_solver, params = define_parts_for_stoch_stability_function.initialize_SDEsolver(params)

    # Create the variables to write output
    output = save_solution.initialize(output, params)

    # Create perturbation
    output = define_perturbation.create_perturbation(params, fenics_params, output)

    # Solve the system
    output = solve_PDE_model.solution_loop(params, output, fenics_params, stoch_solver, u_n, v_n, T_n, k_n)

    # plot_solution.make_3d_plot(output, params, fenics_params)

    # Write the solution to a h5 file
    save_solution.save_solution(
        output,
        params,
        fenics_params,
        f"uG_{params.u_G}_perturbstr_{params.perturbation_strength}_sim_{params.sim_index}",
    )

def setup_simulation_parameters(input_val, parameter_class_val):
    if not isinstance(input_val, np.ndarray):
        parameter_class_val.u_G = input_val
    else:
        parameter_class_val.perturbation_param = input_val[0]
        parameter_class_val.perturbation_type = input_val[1]
        parameter_class_val.perturbation_time_spread = int(input_val[2])
        parameter_class_val.perturbation_height_spread = int(input_val[3])
        parameter_class_val.u_G = float(input_val[4])
        parameter_class_val.perturbation_strength = float(input_val[5])
        parameter_class_val.sim_index = float(input_val[6])

    return parameter_class_val


def run_single_simulation_model(
        model_param,
        fenics_param,
        output_val,
        sim_parameters,
):
    """Function to perform one single model run for a given set of parameters."""
    # Update parameter
    model_param = setup_simulation_parameters(sim_parameters, model_param)

    # Define file name for initial conditions
    model_param.init_path = f'{output_val.init_directory}{model_param.init_cond_path}Ug{model_param.u_G}'

    # Creat sub directory for current simulation type
    output_val = save_solution.create_sub_solution_directory(model_param, output_val)

    # Check if initial condition file exists
    if model_param.load_ini_cond:
        try:
            np.load(model_param.init_path + "_u.npy")
        except Exception as e:
            print(e)
            return

    # Save parameter specifications in file for later reference
    if model_param.sim_index == 0:
        save_solution.save_parameters_in_file(
            model_param,
            output_val,
            file_spec=f"uG_{model_param.u_G}_perturbstr_{model_param.perturbation_strength}",
        )

    # Solve model
    combine_model_solver_functions(fenics_param, model_param, output_val)


def setup_for_sensitivity_study(parameters):
    if parameters.perturbation_param == 'u and theta' or parameters.perturbation_param == 'theta and u':
        perturbation_param_list = ['u', 'theta']
    else:
        perturbation_param_list = [parameters.perturbation_param]

    if parameters.perturbation_type == 'pos and neg' or parameters.perturbation_type == 'neg and pos':
        perturbation_type_list = ['pos', 'neg']
    else:
        perturbation_type_list = [parameters.perturbation_type]

    if parameters.perturbation_time_spread=='all':
        perturbation_time_spread_list = np.arange(100, 500, 100)
    else:
        perturbation_time_spread_list = [parameters.perturbation_time_spread]

    if parameters.perturbation_height_spread=='all':
        perturbation_height_spread_list = np.array([1, 5, 10])
    else:
        perturbation_height_spread_list = [parameters.perturbation_height_spread]

    u_G_list = np.round(parameters.u_G_range, 1)

    perturb_strength_list = np.round(np.arange(0, parameters.perturbation_max, parameters.perturbation_step_size), 4)

    sim_idx_list = np.arange(0,parameters.num_simulation).astype(int)

    param_combination = np.array(np.meshgrid(perturbation_param_list,
                                             perturbation_type_list,
                                             perturbation_time_spread_list,
                                             perturbation_height_spread_list,
                                             u_G_list,
                                             perturb_strength_list,
                                             sim_idx_list)
                                 ).T.reshape(-1, 7)

    return param_combination


def split_into_job_array_tasks(param_comb):
    if len(sys.argv) > 1:
        job_idx = int(sys.argv[1]) - 1
        task_indices = np.linspace(0, np.shape(perturb_param_comb)[0], int(sys.argv[2])).astype(int)
        if task_indices[-1] != np.shape(perturb_param_comb)[0]:
            task_indices = np.append(task_indices, np.shape(perturb_param_comb)[0])
        perturb_param_comb = perturb_param_comb[task_indices[job_idx]:task_indices[job_idx + 1]]


def run_sensitivity_study(in_params, fen_params, out_params):
    """Function to run model for a combination of parameters in parallel."""
    # Make sure that all required parameter are given
    if in_params.perturbation_type == 'none' or in_params.perturbation_param == 'none':
        sys.exit("The type of perturbation and to which equation it should be added needs to be specified to run the "
                 "sensitivity analysis.")
    # Define range of parameters
    unique_param_combinations = [setup_for_sensitivity_study(in_params)[0]]

    # Split parameter combination list into blocks such that each job in a job array has roughly the same amount
    # of tasks
    #unique_param_combinations = split_into_job_array_tasks(unique_param_combinations)

    # Solve model for every parameter combination
    with multiprocessing.Pool(processes=in_params.num_proc) as pool:
        pool.map(partial(run_single_simulation_model, in_params, fen_params, out_params), unique_param_combinations)


def run_multi_uG_simulations(in_params, fen_params, out_params):
    """Function to run model for a several geostrophic wind in parallel."""
    with multiprocessing.Pool(processes=in_params.num_proc) as pool:
        pool.map(partial(run_single_simulation_model, in_params, fen_params, out_params), in_params.u_G_range)


def run_model():
    """Main function to run model depending on user specification"""
    input_params, fenics_params, output_params = make_setup_for_model_run()

    if input_params.sensitivity_study:
        run_sensitivity_study(input_params, fenics_params, output_params)
    elif input_params.u_G_range is not None:
        run_multi_uG_simulations(input_params, fenics_params, output_params)
    else:
        run_single_simulation_model(input_params, fenics_params, output_params)

    return output_params.solution_directory
