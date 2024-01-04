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

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for row_idx in data.index:
        ekman_height_idx[0, row_idx] = np.argmax(
            np.isclose(data.iloc[row_idx, :], u_G, atol=1e-1)
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


def make_line_plot(data, save_dir):

    columns = data.columns.astype(float)

    sm = plt.cm.ScalarMappable(
        cmap="cmc.batlow",
        norm=plt.Normalize(vmin=columns.min(), vmax=columns.max()),
    )
    sm._A = []

    #data = data.rolling(10*60).mean()

    data.plot(kind="line", colormap="cmc.batlow", legend=False, figsize=(10, 5))

    cbar = plt.colorbar(sm)
    cbar.set_label(r"$s_G$ [m/s]")#, rotation=0, labelpad=50)
    plt.ylabel('Ekman layer height [m]')
    plt.xlabel(r't [h]')
    plt.tight_layout()

    plt.savefig(save_dir + 'ekman_layer_height_line_plot.png')


def make_box_plot(data, save_dir):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="variable", y="value", data=pd.melt(data), palette="cmc.batlow")
    data.iloc[40*60].plot(kind='line')
    plt.axhline(20, color='red', linestyle='--', lw=2)
    plt.ylabel('Ekman layer height [m]')
    plt.xlabel(r'$s_G$ [m/s]')
    plt.tight_layout()

    plt.savefig(save_dir + 'ekman_layer_height_box_plot.png')


df_ekman_layer_height = collect_ekman_layer_height_for_all_uG(sim_data_directory)

#make_line_plot(df_ekman_layer_height, vis_data_directory)
#make_box_plot(df_ekman_layer_height, vis_data_directory)

def collect_wind_speed_for_all_uG(directory, height=20):
    sol_file_paths = get_all_data_files(directory)
    u_G_list, solution_files_grouped = group_solution_files_by_uG(sol_file_paths)

    dic_wind_speed = {}

    for key, value in solution_files_grouped.items():
        if float(key) <= 2.9:
            with h5py.File(value[0], "r+") as file:
                u = file["u"][:]
                v = file["v"][:]
                z = file["z"][:]
                t = file["t"][:]
                z_idx = (np.abs(z - height)).argmin()
            wind_speed = np.sqrt(u[z_idx, :] ** 2 + v[z_idx, :] ** 2)
            dic_wind_speed[key] = wind_speed

    dic_wind_speed = pd.DataFrame({k: list(v) for k, v in dic_wind_speed.items()})
    dic_wind_speed = dic_wind_speed.reindex(sorted(dic_wind_speed.columns), axis=1)

    # Drop ten hours
    ten_h_idx = 10

    return dic_wind_speed[ten_h_idx:]

def make_box_plot_wind_speed(data, save_dir):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="variable", y="value", data=pd.melt(data), palette="cmc.batlow")
    #plt.axhline(20, color='red', linestyle='--', lw=2)
    plt.ylabel('s')
    plt.xlabel(r'$s_G$ [m/s]')
    plt.tight_layout()

    plt.savefig(save_dir + 'wind_speed_box_plot.png')

wind_speed_data = collect_wind_speed_for_all_uG(sim_data_directory, height=20)
make_box_plot_wind_speed(wind_speed_data, vis_data_directory)