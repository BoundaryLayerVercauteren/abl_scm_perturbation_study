import cmcrameri.cm as cram
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots

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

data_directory = 'results/long_tail/dt_1/'


def get_all_data_files(main_path):
    solution_file_paths = []
    for path, subdirs, files in os.walk(main_path):
        if 'pos_theta' in path:
            for name in files:
                if 'solution_uG_1.0_perturbstr_0.001_sim_0.h5' in name:
                    solution_file_paths.append(os.path.join(path, name))

    return solution_file_paths


def get_perturbation_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        t = file["t"][:]
        perturbation = file["perturbation"][:]

    T, Z = np.meshgrid(t, z)

    return perturbation, T, Z


data_file_paths = np.unique(np.array(get_all_data_files(data_directory)))

fig, ax = plt.subplots(5, 3, figsize=(25, 15), sharex=True, sharey=True)
ax = ax.ravel()
for idx, file in enumerate(data_file_paths):
    X, Y, data = get_perturbation_data(file)
    ax[idx].contourf(X, Y, data, cmap=cram.lapaz)
    ax[idx].set_title(data_file_paths.split('/')[3])

# ax.set_xlim((0, 1))
# ax.set_ylim((0, 50))
#
# ax.set_xlabel("t [h]")
# ax.set_ylabel("z [m]")
# fig.colorbar(cp)

plt.savefig(data_directory + 'perturbations.png', bbox_inches="tight", dpi=300)

# To clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close("all")  # Closes all the figure windows.
