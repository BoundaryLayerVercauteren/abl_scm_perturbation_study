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


def make_setup_for_model_run():
    # Initialize parameters
    params, fparams, output = parameters.initialize_project_variables()

    # To make naming of output files correct
    if params.perturbation_type == 'none':
        params.perturbation_strength = 'nan'

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


def run_single_simulation_model(
        model_param,
        fenics_param,
        output_val,
        u_G_param=None,
        perturb_param=None,
        sim_index=0,
):
    """Function to perform one single model run for a given set of parameters."""
    # Update parameter
    model_param.sim_index = sim_index
    if u_G_param is not None and len(u_G_param) > 1:
        model_param.u_G = u_G_param[0]
        model_param.perturbation_strength = u_G_param[1]
        if len(u_G_param) > 2:
            model_param.sim_index = u_G_param[2]
    else:
        if u_G_param is not None:
            model_param.u_G = np.around(u_G_param, 1)
        if perturb_param is not None:
            model_param.perturbation_strength = np.around(perturb_param, 3)
        elif model_param.perturbation_param is not 'none':
            model_param.perturbation_strength = np.around(model_param.perturbation_max, 3)
    # Define file name for initial conditions
    model_param.init_path = (
            output_val.init_directory
            + model_param.init_cond_path
            + "Ug"
            + str(model_param.u_G)
    )

    # Check if initial condition file exists
    if model_param.load_ini_cond:
        try:
            np.load(model_param.init_path + "_u.npy")
        except Exception as e:
            print(e)
            return

    # Save parameter specifications in file for later reference
    if sim_index == 0:
        save_solution.save_parameters_in_file(
            model_param,
            output_val,
            file_spec=f"uG_{model_param.u_G}_perturbstr_{model_param.perturbation_strength}",
        )

    # Solve model
    combine_model_solver_functions(fenics_param, model_param, output_val)


def run_sensitivity_study(in_params, fen_params, out_params):
    """Function to run model for a combination of parameters in parallel."""
    # Make sure that all required parameter are given
    if in_params.perturbation_type == 'none' or in_params.perturbation_param == 'none':
        sys.exit("The type of perturbation and to which equation it should be added needs to be specified to run the "
                 "sensitivity analysis.")
    # Define range of parameters for geostrophic wind and strength of the perturbation
    u_G_range = np.round(in_params.u_G_range, 1)
    perturb_strength_list = np.round(np.arange(0, in_params.perturbation_max, in_params.perturbation_step_size), 4)
    if in_params.num_simulation == 1:
        unique_param_combinations = np.array(np.meshgrid(u_G_range, perturb_strength_list)).T.reshape(-1, 2)
    else:
        sim_range = np.arange(0, in_params.num_simulation, 1)
        unique_param_combinations = np.array(np.meshgrid(u_G_range, perturb_strength_list, sim_range)).T.reshape(-1, 3)

    # if len(sys.argv) > 1:
    #     job_idx = int(sys.argv[1]) - 1
    #     if in_params.num_simulation > 1:
    #         task_indices = np.arange(0, in_params.num_simulation + in_params.num_proc, in_params.num_proc)
    #     else:
    #         task_indices = np.arange(0, np.shape(unique_param_combinations)[0], in_params.num_proc)
    #         if task_indices[-1] != np.shape(unique_param_combinations)[0]:
    #             task_indices = np.append(task_indices, np.shape(unique_param_combinations)[0])
    #     unique_param_combinations = unique_param_combinations[task_indices[job_idx]:task_indices[job_idx + 1]]

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

    if input_params.sensitivity_study and input_params.perturbation_param=='pde_all':
        perturb_param = ['pde_u', 'pde_theta']
        perturb_type = ['pos_gaussian', 'neg_gaussian']
        if input_params.perturbation_time_spread == 'grid':
            perturbation_time_spread = np.array([1,5,10])
            perturbation_height_spread = np.arange(100, 500, 100)
        else:
            perturbation_time_spread = input_params.perturbation_time_spread
            perturbation_height_spread = input_params.perturbation_height_spread

        perturb_param_comb = np.array(np.meshgrid(perturb_param,
                                                  perturb_type,
                                                  perturbation_time_spread,
                                                  perturbation_height_spread)).T.reshape(-1, 4)
        if len(sys.argv) > 1:
            job_idx = int(sys.argv[1]) - 1
            task_indices = np.arange(0, np.shape(perturb_param_comb)[0], int(sys.argv[2]))
            if task_indices[-1] != np.shape(perturb_param_comb)[0]:
                task_indices = np.append(task_indices, np.shape(perturb_param_comb)[0])
            perturb_param_comb = perturb_param_comb[task_indices[job_idx]:task_indices[job_idx + 1]]

            for elem in perturb_param_comb:
                input_params, fenics_params, output_params = make_setup_for_model_run()
                input_params.perturbation_param = elem[0]
                input_params.perturbation_type = elem[1]
                input_params.perturbation_time_spread = int(elem[2])
                input_params.perturbation_height_spread = int(elem[3])
                run_sensitivity_study(input_params, fenics_params, output_params)
    else:
        if input_params.sensitivity_study:
            run_sensitivity_study(input_params, fenics_params, output_params)
        elif input_params.u_G_range is not None:
            run_multi_uG_simulations(input_params, fenics_params, output_params)
        else:
            run_single_simulation_model(input_params, fenics_params, output_params)

    return output_params.solution_directory
