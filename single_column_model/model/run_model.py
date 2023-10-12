import multiprocessing
import sys
from functools import partial
from itertools import product

import numpy as np

if "pytest" in sys.modules:
    from single_column_model.tests import parameters
else:
    from single_column_model.model import parameters

# Project related imports
from single_column_model.model import \
    space_discretization  # define_stochastic_part,
from single_column_model.model import (define_initial_and_boundary_conditions,
                                       define_PDE_model, define_perturbation,
                                       define_stochastic_part, solve_PDE_model)
from single_column_model.utils import plot_solution, save_solution


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
    mesh, params = space_discretization.create_grid(params, "power")

    # Create stochastic grid
    #params.stoch_grid, params.Hs_det_idx = define_stochastic_part.make_stochastic_grid(params, mesh.coordinates())

    # Define variables to use the fenics library
    fenics_params = define_PDE_model.setup_fenics_variables(fenics_params, mesh)

    # Define boundary conditions
    fenics_params, params = define_initial_and_boundary_conditions.define_boundary_conditions(fenics_params, params)

    # Define initial profiles/ values
    u_n, v_n, T_n, k_n = define_initial_and_boundary_conditions.define_initial_conditions(fenics_params.Q, mesh, params)

    # Set up the weak formulation of the equations
    fenics_params.F = define_PDE_model.weak_formulation(fenics_params, params, u_n, v_n, T_n, k_n)

    # Create the variables to write output
    output = save_solution.initialize(output, params)

    # Create perturbation
    output = define_perturbation.create_perturbation(params, fenics_params, output)

    # Solve the system
    output = solve_PDE_model.solution_loop(params, output, fenics_params, u_n, v_n, T_n, k_n)

    #plot_solution.make_3d_plot(output, params, fenics_params)

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
    if u_G_param is not None:
        model_param.u_G = np.around(u_G_param, 1)
    if perturb_param is not None:
        model_param.perturbation_strength = np.around(perturb_param, 3)

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
    u_G_range = in_params.u_G_range
    perturb_strength_list = np.round(np.arange(0, 0.03, 0.001), 3)
    # Create parameter grid, i.e. all combinations of u_G and perturbation strength
    unique_param_combinations = list(
        list(zip(u_G_range, element))
        for element in product(perturb_strength_list, repeat=len(u_G_range))
    )
    unique_param_combinations = [elem for sublist in unique_param_combinations for elem in sublist]

    # Solve model for every parameter combination
    with multiprocessing.Pool(processes=in_params.num_proc) as pool:
        for param_val in unique_param_combinations:
            pool.map(partial(run_single_simulation_model, in_params, fen_params, out_params, param_val[0],
                             param_val[1]), range(in_params.num_simulation))


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
