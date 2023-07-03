"""Main file to combine all functions related to solve atmospheric boundary single-column model."""

# project related imports
from single_column_model.model import initial_and_boundary_conditions as ic
from single_column_model.model import define_PDE_model as fut
from single_column_model.model import space_discretization as sd
from single_column_model.utils import save_solution as ss
from single_column_model.model import solve_PDE_model as ut
from single_column_model.model import define_perturbation as dp


def solve_turb_model(fenics_params, params, output):
    
    # create mesh; options: 'power' , 'log', 'log_lin'
    mesh, params = sd.create_grid(params, 'power', show=False)
    
    # define variables to use the fenics lib
    fenics_params = fut.setup_fenics_variables(fenics_params, mesh)
    
    # define boundary conditions
    fenics_params, params = ic.def_boundary_conditions(fenics_params, params)
    
    # define initial profiles/ values
    u_n, v_n, T_n, k_n = ic.def_initial_conditions(fenics_params.Q, mesh, params)
    
    # set up the weak formulation of the equations
    fenics_params.F = fut.weak_formulation(fenics_params, params, u_n, v_n, T_n, k_n)
    
    # create the variables to write output
    output = ss.initialize(output, params)
    
    # define the solver and its parameters
    solver = fut.prepare_fenics_solver(fenics_params, fenics_params.F)

    # Create perturbation
    output = dp.create_perturbation(params, fenics_params, output)
    
    # solve the system
    output = ut.solution_loop(solver, params, output, fenics_params, u_n, v_n, T_n, k_n)
    
    # write the solution to a h5 file
    ss.save_solution(output, params, fenics_params, str(params.u_G) + '_sim' + str(params.sim_index))