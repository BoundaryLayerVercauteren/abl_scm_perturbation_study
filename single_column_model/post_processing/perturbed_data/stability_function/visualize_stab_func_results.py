import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scienceplots
import cmcrameri.cm as cmc

# Set plotting style
plt.style.use("science")

# Set font sizes for plots
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define directory where simulation output is saved
output_directory = (
    "results/long_tail/stab_func/gauss_process_stab_func/negative/2.4/1_0/"
)

# Get all solution files
output_files = []
for path, subdirs, files in os.walk(output_directory):
    for name in files:
        if "solution" in name:
            output_files.append(os.path.join(path, name))


def combine_all_sim_files(file_paths):
    richardson_dict = {}
    delta_theta_data_dict = {}

    for file_path in file_paths:
        sim_idx = int(float(file_path.split(".")[-3].split("_")[-1]))
        with h5py.File(file_path, "r+") as file:
            z = file["z"][:]
            z_idx = (np.abs(z - 20)).argmin()
            theta = file["theta"][:]
            delta_theta_data_dict[sim_idx] = theta[z_idx, :] - theta[0, :]
            u = file["u"][:]
            v = file["v"][:]
            richardson_dict[sim_idx] = (
                (9.81 / 300)
                * ((theta[z_idx, :] - theta[0, :]) / 20)
                / (
                    ((u[z_idx, :] - u[0, :]) / 20) ** 2
                    + ((v[z_idx, :] - v[0, :]) / 20) ** 2
                )
            )
            t = file["t"][:]

    delta_theta_data = pd.DataFrame.from_dict(delta_theta_data_dict)
    richardson_data = pd.DataFrame.from_dict(richardson_dict)

    delta_theta_data = delta_theta_data.reindex(
        sorted(delta_theta_data.columns), axis=1
    )
    richardson_data = richardson_data.reindex(sorted(richardson_data.columns), axis=1)

    richardson_data = richardson_data.set_index(t.flatten())
    delta_theta_data = delta_theta_data.set_index(t.flatten())

    return delta_theta_data, richardson_data


data_delta_theta, data_richardson = combine_all_sim_files(output_files)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# data_richardson.plot(ax=ax[0], kind="line", colormap="gray_r", legend=False)
# data_richardson.mean(axis=1).plot(ax=ax[0], kind="line", color="red", legend=False, marker='s', markevery=60)

data_delta_theta.plot(ax=ax, kind="line", colormap="gray_r", legend=False)
data_delta_theta.mean(axis=1).plot(
    ax=ax, kind="line", color="red", legend=False, marker="s", markevery=60
)

# ax[0].set_xlabel('time [h]')
# ax[1].set_xlabel('time [h]')
ax.set_xlabel("time [h]")

# ax[0].set_ylabel("Ri")
# ax[1].set_ylabel(r"$\Delta \theta$ [K]")
ax.set_ylabel(r"$\Delta \theta$ [K]")

plt.savefig(output_directory + "transitions.png", bbox_inches="tight", dpi=300)
