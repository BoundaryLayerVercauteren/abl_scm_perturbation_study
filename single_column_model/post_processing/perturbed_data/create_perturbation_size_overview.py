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
        if '/pos_theta/simulations' in path:
            for name in files:
                if 'solution_uG_1.0_perturbstr_0.001_sim_0.h5' in name:
                    solution_file_paths.append(os.path.join(path, name))

    solution_file_paths = np.unique(np.array(solution_file_paths))

    grid_dirs = ['100_1/', '100_5/', '100_10/',
                 '200_1/', '200_5/', '200_10/',
                 '300_1/', '300_5/', '300_10/',
                 '400_1/', '400_5/', '400_10/',
                 '500_1/', '500_5/', '500_10/']

    sorted_solution_file_paths = np.copy(solution_file_paths)

    for idx, grid_size in enumerate(grid_dirs):
        for file in solution_file_paths:
            if grid_size in file:
                sorted_solution_file_paths[idx]=file

    return sorted_solution_file_paths


def get_perturbation_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        t = file["t"][:]
        perturbation = file["perturbation"][:]

    T, Z = np.meshgrid(t, z)

    return perturbation, T, Z


data_file_paths = get_all_data_files(data_directory)
print(data_file_paths)
fig, ax = plt.subplots(5, 3, figsize=(25, 15), sharex=True, sharey=True)
ax = ax.ravel()
for idx, file in enumerate(data_file_paths):
    try:
        print(idx, file)
        data, X, Y = get_perturbation_data(file)
        im = ax[idx].contourf(X, Y, data, cmap=cram.lapaz)
        ax[idx].set_xlim((0, 1))
        ax[idx].set_ylim((0, 50))
        if idx == 0 or idx == 1 or idx == 2:
            ax[idx].set_title(rf'$z_s={file.split("/")[3].split("_")[1]}$m')
        if idx==2 or idx==5 or idx==8:
            ax[idx].annotate(rf'$t_s={file.split("/")[3].split("_")[0]}$m', xy=(1.1, 0.5), rotation=90,
                       ha='center', va='center', xycoords='axes fraction')
        if idx == 6 or idx == 7 or idx == 8:
            ax[idx].tick_params(axis='x', rotation=45)
    except Exception:
        pass

fig.text(0.5, 0.08, 't [h]', ha='center')
fig.text(0.1, 0.5, 'z [m]', va='center', rotation='vertical')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("r", rotation=0)

plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.savefig(data_directory + 'perturbations.png', bbox_inches="tight", dpi=300)

# To clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close("all")  # Closes all the figure windows.
