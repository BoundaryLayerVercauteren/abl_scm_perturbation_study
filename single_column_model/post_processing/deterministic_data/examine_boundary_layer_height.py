"""Run with python3 -m single_column_model.post_processing.deterministic_data.examine_boundary_layer_height"""

import h5py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cram
import seaborn as sns

from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(10,10))

data_directory = 'single_column_model/solution/short_tail/deterministic/'
sim_data_directory = data_directory + 'simulations/'
vis_data_directory = data_directory + 'visualization/'


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


def find_Ekman_layer_height(file_path, u_G):
    # Open output file and load variables
    with h5py.File(file_path, "r+") as file:
        u = file["u"][:]
        v = file["v"][:]
        z = file["z"][:]
        t = file["t"][:]

    wind_speed = np.sqrt(u**2+v**2)

    data = pd.DataFrame(data=wind_speed.T, columns=z.flatten())

    lower_z = [col for col in data.columns if col < 5]
    data.loc[:, lower_z] = np.nan

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for row_idx in data.index:
        ekman_height_idx[0, row_idx] = np.argmax(
            np.isclose(data.iloc[row_idx, :], u_G, atol=2e-1)
        )

    return z[list(map(int, ekman_height_idx.flatten())), :].flatten(), t.flatten()


def collect_ekman_layer_height_for_all_uG(directory):
    sol_file_paths = get_all_data_files(directory)
    u_G_list, solution_files_grouped = group_solution_files_by_uG(sol_file_paths)

    dic_ekman_layer = {}

    for key, value in solution_files_grouped.items():
        if float(key) <= 2.9:
            dic_ekman_layer[key], time = find_Ekman_layer_height(value[0], float(key))

    dic_ekman_layer['time'] = time

    df_ekman_layer_height = pd.DataFrame({k: list(v) for k, v in dic_ekman_layer.items()})
    df_ekman_layer_height = df_ekman_layer_height.reindex(sorted(df_ekman_layer_height.columns), axis=1)

    df_ekman_layer_height = df_ekman_layer_height.set_index("time")

    # Drop ten hours
    ten_h_idx = 10

    return df_ekman_layer_height[ten_h_idx:]


def make_line_plot(data, axes):

    columns = data.columns.astype(float)

    sm = plt.cm.ScalarMappable(cmap="cmc.batlow", norm=plt.Normalize(vmin=columns.min(), vmax=columns.max()))
    sm._A = []

    data.plot(kind="line", colormap="cmc.batlow", legend=False, ax=axes)

    cbar = plt.colorbar(sm)
    cbar.set_label(r"$s_G \ [\mathrm{ms^{-1}}]$")
    axes.set_ylabel('Ekman layer height [m]')
    axes.set_xlabel(r't [h]')
    axes.set_title('a)', loc='left')


def make_box_plot(data, axes):
    sns.boxplot(x="variable", y="value", data=pd.melt(data), palette="cmc.batlow", ax=axes)
    axes.axhline(20, color='red', linestyle='--', lw=2)
    axes.set_ylabel('Ekman layer height [m]')
    axes.set_xlabel(r'$s_G \ [\mathrm{ms^{-1}}]$')
    axes.locator_params(axis='x', nbins=10)
    axes.set_title('b)', loc='left')


df_ekman_layer_height = collect_ekman_layer_height_for_all_uG(sim_data_directory)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax = ax.ravel()

make_line_plot(df_ekman_layer_height, ax[0])
make_box_plot(df_ekman_layer_height, ax[1])

fig.tight_layout()
plt.savefig(vis_data_directory + 'ekman_layer_height.png', bbox_inches="tight", dpi=300)