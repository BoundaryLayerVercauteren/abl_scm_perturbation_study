# standard imports
from dataclasses import dataclass
import matplotlib.pyplot as plt
from numpy.random import rand
from tqdm import tqdm
from fenics import *
import numpy as np
import runpy
import h5py
import os

# project related imports
from single_column_model.model import initial_and_boundary_conditions as ic
from single_column_model.model import solve_PDE_model as fut
from single_column_model.model import space_discretization as sd
from single_column_model.utils import save_solution as ss
from single_column_model.model import utility_functions as ut
from single_column_model.model import parameters as pc


def solve_turb_model(fparams, params, output):
    
    # create mesh; options: 'power' , 'log', 'log_lin'
    mesh, params = sd.create_grid(params, 'power', show=False)
    
    # define variables to use the fenics lib
    fparams = fut.setup_fenics_variables(fparams, mesh)
    
    # define boundary conditions
    fparams, params = ic.def_boundary_conditions(fparams, params)
    
    # define initial profiles
    u_n, v_n, T_n, k_n = ic.def_initial_cnditions(fparams.Q, mesh, params)
    
    # setup the weak formulation of the equations
    F = fut.weak_formulation(fparams, params, u_n, v_n, T_n, k_n)
    
    # create the variables to write output
    output = ss.initialize(output, params)
    
    # define the solver and its parameters
    solver = fut.prepare_fenics_solver(fparams, F)
    
    # solve the system
    output = ut.solution_loop(solver, params, output, fparams, u_n, v_n, T_n, k_n)
    
    # write the solution to h5 file
    ss.save_solution(output, params, fparams, str(params.u_G) + '_sim' + str(params.sim_index))