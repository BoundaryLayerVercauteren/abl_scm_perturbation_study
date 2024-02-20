import cmcrameri.cm as cram
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots

plt.style.use("science")

# set font sizes for plots
SMALL_SIZE = 18 * 1.5
MEDIUM_SIZE = 22 * 1.5
BIGGER_SIZE = 30 * 1.5

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

directory_path = "results/long_tail/dt_1/500_10/"
perturb_dir = ["neg_theta/", "pos_theta/", "neg_u/", "pos_u/"]
perturb_strengths = ["0.009", "0.008", "0.006", "0.007"]
uGs = ["1.0", "1.7", "1.8", "2.5"]
param_comb = ["1.0", "neg_theta/"]


def get_all_data_files(path):
    solution_file_paths = []
    for path, subdirs, files in os.walk(path):
        if "simulations" in path:
            for name in files:
                if "solution" in name:
                    solution_file_paths.append(os.path.join(path, name))

    return solution_file_paths


# Build paths
data_paths = get_all_data_files(directory_path)
filtered_data_paths = []

for path in data_paths:
    for idx in np.arange(0, len(uGs)):
        if f"perturbstr_{perturb_strengths[idx]}" in path and f"uG_{uGs[idx]}" in path:
            filtered_data_paths.append(path)


def get_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        t = file["t"][:]
        u = file["u"][:]
        v = file["v"][:]
        wind_speed = np.sqrt(u ** 2 + v ** 2)
        theta = file["theta"][:]

    return wind_speed, theta, t, z


fig, ax = plt.subplots(4, 3, figsize=(20, 15), sharex=True, sharey=True)
ax = ax.ravel()
ax_idx = 0

for path in filtered_data_paths:
    wind_speed, theta, t, z = get_data(path)
    X, Y = np.meshgrid(t, z)
    ax[ax_idx].pcolor(X, Y, wind_speed, cmap=cram.davos)
    ax[ax_idx + 1].pcolor(X, Y, theta, cmap=cram.lajolla)
