import os

import h5py
import numpy as np
import pandas as pd

data_directory = 'single_column_model/solution/short_tail/perturbed/pde_theta/neg/'
sim_directory = data_directory + 'simulations/'


def get_all_data_files(path):
    solution_file_paths = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if 'solution' in name:
                solution_file_paths.append(os.path.join(path, name))

    return solution_file_paths


def group_solution_files_by_uG(file_list):
    solution_files_grouped = {}
    u_G_list = []
    for file in file_list:
        file_name = file.split('/')[-1]
        u_G_list.append(file_name.split('_')[2])
    u_G_list = np.unique(u_G_list)

    for uG in u_G_list:
        solution_files_grouped[uG] = []

    for file in file_list:
        for uG in u_G_list:
            if f'uG_{uG}' in file:
                solution_files_grouped[uG].append(file)

    return u_G_list, solution_files_grouped


def get_temp_inversion_data(all_file_paths, z_idx=37):
    df_delta_theta_temp = {}

    for file_path in all_file_paths:

        try:
            with h5py.File(file_path, "r+") as file:
                t = file["t"][:]
                # Set name of column
                r = file["r"][:][0][0]
                # Calculate temperature inversion
                theta = file["theta"][:]

                if any(theta == np.nan):
                    print(file_path)
                else:
                    df_delta_theta_temp[r] = theta[z_idx, :] - theta[0, :]

        except Exception:
            # print(file_path)
            continue

    print(df_delta_theta_temp)
    df_delta_theta = pd.DataFrame({k: list(v) for k, v in df_delta_theta_temp.items()})
    print(df_delta_theta)
    df_delta_theta = df_delta_theta.reindex(sorted(df_delta_theta.columns), axis=1)
    df_delta_theta["time"] = t.flatten()

    return df_delta_theta


def calculate_num_sim_with_transition(data):
    columns_with_transitions = []
    for idx, column in enumerate(data.columns):
        if column != 'time' and data.iloc[0, idx] >= 5 and (data[column] < 5).any():
            columns_with_transitions.append(column)
        elif column != 'time' and data.iloc[0, idx] < 5 and (data[column] > 5).any():
            columns_with_transitions.append(column)

    if len(columns_with_transitions) > 0:
        return min(columns_with_transitions)
    else:
        return np.nan


def get_transition_statistics(grouped_file_paths):
    min_perturb_strength = {}
    for key, _ in grouped_file_paths.items():
        data = get_temp_inversion_data(grouped_file_paths[key])
        min_perturb_strength[key] = calculate_num_sim_with_transition(data)

    print(min_perturb_strength)


solution_files = get_all_data_files(sim_directory)
uGs, solution_files_uG = group_solution_files_by_uG(solution_files)

get_transition_statistics(solution_files_uG)
