import h5py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cram
import seaborn as sns

plt.style.use("science")

# set font sizes for plots
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_directory = 'single_column_model/solution/long_tail/deterministic/'
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
        z = file["z"][:]
        t = file["t"][:]

    data = pd.DataFrame(data=u.T, columns=z.flatten())

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for row_idx in data.index:
        ekman_height_idx[0, row_idx] = np.argmax(
            np.around(data.iloc[row_idx, :], 1) == u_G
        )

    return z[list(map(int, ekman_height_idx.flatten())), :].flatten(), t.flatten()


def collect_ekman_layer_height_for_all_uG(directory):
    sol_file_paths = get_all_data_files(directory)
    u_G_list, solution_files_grouped = group_solution_files_by_uG(sol_file_paths)

    dic_ekman_layer = {}

    for key, value in solution_files_grouped.items():
        if float(key) <= 3.2:
            dic_ekman_layer[key], time = find_Ekman_layer_height(value[0], float(key))

    dic_ekman_layer['time'] = time

    df_ekman_layer_height = pd.DataFrame({k: list(v) for k, v in dic_ekman_layer.items()})
    df_ekman_layer_height = df_ekman_layer_height.reindex(sorted(df_ekman_layer_height.columns), axis=1)

    df_ekman_layer_height = df_ekman_layer_height.set_index("time")

    # Drop ten hours
    ten_h_idx = 10

    return df_ekman_layer_height[ten_h_idx:]


def make_line_plot(data, save_dir):

    columns = data.columns.astype(float)

    sm = plt.cm.ScalarMappable(
        cmap="cmc.batlow",
        norm=plt.Normalize(vmin=columns.min(), vmax=columns.max()),
    )
    sm._A = []

    data.plot(kind="line", colormap="cmc.batlow", legend=False, figsize=(10, 5))

    cbar = plt.colorbar(sm)
    cbar.set_label(r"$u_G$ [m/s]")#, rotation=0, labelpad=50)
    plt.ylabel('Ekman layer height [m]')
    plt.xlabel(r't [h]')
    plt.tight_layout()

    plt.savefig(save_dir + 'ekman_layer_height_line_plot.png')


def make_box_plot(data, save_dir):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="variable", y="value", data=pd.melt(data), palette="cmc.batlow")
    plt.ylabel('Ekman layer height [m]')
    plt.xlabel(r'$u_G$ [m/s]')
    plt.tight_layout()

    plt.savefig(save_dir + 'ekman_layer_height_box_plot.png')


df_ekman_layer_height = collect_ekman_layer_height_for_all_uG(sim_data_directory)

make_line_plot(df_ekman_layer_height, vis_data_directory)
make_box_plot(df_ekman_layer_height, vis_data_directory)