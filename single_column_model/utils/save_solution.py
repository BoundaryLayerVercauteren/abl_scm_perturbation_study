#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:54:19 2021

@author: slava
"""

# standard imports
import matplotlib.pyplot as plt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py
import time
import os
import json

# project related imports
from single_column_model.model import define_PDE_model as fut
from single_column_model.model import solve_PDE_model as ut


def create_solution_directory(params):
    """
    Create directory to store solution
    :param output: class to store output
    :return: updated output class
    """
    # Get current date and time
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # Define name of directory where solutions will be stored
    solution_directory = "single_column_model/solution/" + timestr + "/"
    # Create directory (if it does not exist already)
    current_directory = os.getcwd()
    sol_directory = os.path.join(current_directory, solution_directory)
    if not os.path.exists(sol_directory):
        os.makedirs(sol_directory)
    print("The solution files and plots are in: " + str(sol_directory))
    if params.save_ini_cond:
        init_directory = os.path.join(
            current_directory, "single_column_model/init_condition/"
        )
        if not os.path.exists(init_directory):
            os.makedirs(init_directory)
        print("The initial condition files are in: " + str(init_directory))
    elif params.load_ini_cond:
        init_directory = os.path.join(
            current_directory, "single_column_model/init_condition/"
        )
    else:
        init_directory = ""
    return sol_directory, init_directory


def create_sub_solution_directory(params, output):
    if (
        params.perturbation_time_spread is not None
        and params.perturbation_height_spread is not None
    ):
        output.solution_directory = (
            output.top_solution_directory
            + f"{params.perturbation_time_spread}_{params.perturbation_height_spread}/"
        )
        if params.perturbation_param is not None:
            output.solution_directory = (
                output.solution_directory
                + f'{params.perturbation_type}_{params.perturbation_param.replace(" ", "_")}/'
            )
    else:
        if params.perturbation_param is not None:
            output.solution_directory = (
                output.top_solution_directory
                + f'{params.perturbation_type}_{params.perturbation_param.replace(" ", "_")}/'
            )
        if params.perturbation_param == "stab_func":
            output.solution_directory += f'{params.u_G}/{str(params.perturbation_strength).replace(".", "_").replace("-", "")}/'

    try:
        os.makedirs(output.solution_directory)
    except FileExistsError:
        # directory already exists
        pass

    return output


def save_parameters_in_file(params, output, file_spec=""):
    """
    Save values in parameter class to file
    :param output: class which includes name of directory where the file will be stored
    :param params: class of parameters
    :return: none
    """
    # Define name of parameter file
    file_name = str(output.solution_directory) + "parameters_" + file_spec + ".json"
    # Transform parameter class to json
    params_json = params.to_json()
    # Save json to file + remove unnecessary characters
    with open(file_name, "w") as file:
        json.dump(json.loads(params_json), file)


def save_current_result(output, params, fparams, i, us, vs, Ts, ks, phi_s, theta_gs):
    # convert 2 numpy array and save
    output.U_save[:, i] = np.flipud(interpolate(us, fparams.Q).vector().get_local())
    output.V_save[:, i] = np.flipud(interpolate(vs, fparams.Q).vector().get_local())
    output.T_save[:, i] = np.flipud(interpolate(Ts, fparams.Q).vector().get_local())
    output.k_save[:, i] = np.flipud(interpolate(ks, fparams.Q).vector().get_local())
    output.phi_stoch[:, i] = np.flipud(
        interpolate(phi_s, fparams.Q).vector().get_local()
    )

    # Write Ri number
    calc_Ri = project(fut.Ri(fparams, params), fparams.Q)
    output.Ri_save[:, i] = np.flipud(calc_Ri.vector().get_local())

    # Write Km
    calc_Km = project(fut.K_m(fparams, params), fparams.Q)
    output.Km_save[:, i] = np.flipud(calc_Km.vector().get_local())

    # Write Kh
    calc_Kh = project(fut.K_h(fparams, params), fparams.Q)
    output.Kh_save[:, i] = np.flipud(calc_Kh.vector().get_local())

    # Write surface temperature
    output.T_g_save[:, i] = theta_gs

    return output


def calc_variables_np(params, fparams, us, vs):
    # convert 2 numpy array and save
    U_save = np.flipud(interpolate(us, fparams.Q).vector().get_local())
    V_save = np.flipud(interpolate(vs, fparams.Q).vector().get_local())

    # Write Kh
    calc_Kh = project(fut.K_h(fparams, params), fparams.Q)
    Kh_save = np.flipud(calc_Kh.vector().get_local())

    return U_save, V_save, Kh_save


def initialize(output, params):
    output.U_save = np.zeros((params.Nz, params.save_num_steps))
    output.V_save = np.zeros((params.Nz, params.save_num_steps))
    output.T_save = np.zeros((params.Nz, params.save_num_steps))
    output.k_save = np.zeros((params.Nz, params.save_num_steps))
    output.Ri_save = np.zeros((params.Nz, params.save_num_steps))
    output.Kh_save = np.zeros((params.Nz, params.save_num_steps))
    output.Km_save = np.zeros((params.Nz, params.save_num_steps))
    output.phi_stoch = np.zeros((params.Nz, params.save_num_steps))
    output.T_g_save = np.zeros((1, params.save_num_steps))

    output.U_save[:] = np.nan
    output.V_save[:] = np.nan
    output.T_save[:] = np.nan
    output.k_save[:] = np.nan
    output.Ri_save[:] = np.nan
    output.Kh_save[:] = np.nan
    output.Km_save[:] = np.nan
    output.phi_stoch[:] = np.nan
    output.T_g_save[:] = np.nan

    return output


def save_solution(output, params, fparams, file_spec=""):
    if params.save_ini_cond:
        np.save(params.init_path + "_u", output.U_save[:, -2])
        np.save(params.init_path + "_v", output.V_save[:, -2])
        np.save(params.init_path + "_theta", output.T_save[:, -2])
        np.save(params.init_path + "_TKE", output.k_save[:, -2])
        print("\n Current solution saved as initial condition")

    saveFile = h5py.File(
        str(output.solution_directory) + "solution_" + file_spec + ".h5", "w"
    )

    U_ds = saveFile.create_dataset(
        "/u", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    V_ds = saveFile.create_dataset(
        "/v", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    T_ds = saveFile.create_dataset(
        "/theta", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    k_ds = saveFile.create_dataset(
        "/TKE", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    Ri_ds = saveFile.create_dataset(
        "/Ri", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    Kh_ds = saveFile.create_dataset(
        "/Kh", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    Km_ds = saveFile.create_dataset(
        "/Km", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    phi_ds = saveFile.create_dataset(
        "/phi", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    perturbation_ds = saveFile.create_dataset(
        "/perturbation", (params.Nz, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    T_g_ds = saveFile.create_dataset(
        "/theta_g", (1, params.save_num_steps), h5py.h5t.IEEE_F64BE
    )
    r_ds = saveFile.create_dataset("/r", (1, 1))

    z_ds = saveFile.create_dataset("/z", (np.size(fparams.z), 1), h5py.h5t.IEEE_F64BE)
    t_ds = saveFile.create_dataset("/t", (1, params.save_num_steps))

    U_ds[...] = output.U_save
    V_ds[...] = output.V_save
    T_ds[...] = output.T_save
    k_ds[...] = output.k_save
    Ri_ds[...] = output.Ri_save
    Kh_ds[...] = output.Kh_save
    Km_ds[...] = output.Km_save
    phi_ds[...] = output.phi_stoch
    T_g_ds[...] = output.T_g_save

    z_ds[...] = fparams.z
    t_ds[...] = np.linspace(0, params.T_end_h, params.save_num_steps)

    r_ds[...] = output.r
    perturbation_ds[...] = output.perturbation[:, :: params.save_dt_sim]

    saveFile.close()
    print("simulation is done")
