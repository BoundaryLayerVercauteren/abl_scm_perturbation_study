import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

data_directory = 'single_column_model/solution/long_tail/perturbed/'
perturb_dir = ['neg_theta/', 'pos_theta/', 'neg_u/', 'pos_u/']
sim_directory = 'simulations/'


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
    r_list = []
    for file in file_list:
        file_name = file.split('/')[-1]
        u_G_list.append(file_name.split('_')[2])
        r_list.append(float(file_name.split('_')[4]))
    u_G_list = np.unique(u_G_list)
    r_list = np.unique(r_list)

    for uG in u_G_list:
        solution_files_grouped[uG] = []

    for file in file_list:
        for uG in u_G_list:
            if f'uG_{uG}' in file:
                solution_files_grouped[uG].append(file)

    return u_G_list, solution_files_grouped, r_list


def calculate_perturbation_strength(variable, cur_r, r_range, height_idx):
    time_idx = 15
    time_variance_data = np.var(variable[:, :time_idx], axis=1)
    # Normalize the variance vector
    normalized_time_variance_data = (time_variance_data - time_variance_data.min()) / (
            time_variance_data.max() - time_variance_data.min()) + 1
    # Normalize the range of perturbation values
    idx_r = np.where(r_range == cur_r)[0]
    normalized_r_range = (r_range - r_range.min()) / (r_range.max() - r_range.min()) + 1
    # Calculate percentage of related to variance
    perturbation_percentage = normalized_r_range[idx_r] / normalized_time_variance_data[height_idx] * 100

    return perturbation_percentage


def get_temp_inversion_data(all_file_paths, r_range, z_idx=37):
    df_delta_theta_temp = {}

    for file_path in all_file_paths:

        try:
            with h5py.File(file_path, "r+") as file:
                t = file["t"][:]
                # Set name of column
                r = file["r"][:][0][0]
                # Calculate temperature inversion
                theta = file["theta"][:]

                if '_theta/' in file_path:
                    perturb_strength = calculate_perturbation_strength(theta, r, r_range, z_idx)
                elif '_u/' in file_path:
                    perturb_strength = calculate_perturbation_strength(file["u"][:], r, r_range, z_idx)

                if (theta == np.nan).any():
                    print(file_path)
                else:
                    df_delta_theta_temp[perturb_strength] = theta[z_idx, :] - theta[0, :]

        except Exception as e:
            print(e)
            continue

    df_delta_theta = pd.DataFrame({k: list(v) for k, v in df_delta_theta_temp.items()})
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
        return np.min(np.abs(columns_with_transitions))
    else:
        return np.nan


def get_transition_statistics(grouped_file_paths, perturb_range):
    min_perturb_strength = {}
    for key, _ in grouped_file_paths.items():
        data = get_temp_inversion_data(grouped_file_paths[key], perturb_range)
        min_perturb_strength[key] = calculate_num_sim_with_transition(data)

    return min_perturb_strength


labels = [r'$\theta^-$', r'$\theta^+$', r'$u^-$', r'$u^+$']
markers = ['v', '^', 's', 'd']

plt.figure(figsize=(10, 5))

for idx, dir in enumerate(perturb_dir):
    directory_path = data_directory + dir + sim_directory

    solution_files = get_all_data_files(directory_path)
    uGs, solution_files_uG, rs = group_solution_files_by_uG(solution_files)
    min_r_for_uG = get_transition_statistics(solution_files_uG, rs)

    plt.plot(list(min_r_for_uG.keys()), list(min_r_for_uG.values()))
    plt.scatter(min_r_for_uG.keys(), min_r_for_uG.values(), label=labels[idx], marker=markers[idx])

plt.ylabel('r', rotation=0)
plt.xlabel(r'$u_G$ [m/s]')
plt.legend()
plt.tight_layout()
plt.savefig(data_directory + 'sensitivity_analysis.png')
