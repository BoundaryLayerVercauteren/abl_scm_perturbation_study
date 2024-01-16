import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scienceplots
import cmcrameri.cm as cmc

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

# Define directory where simulation output is saved
output_directory = "results/short_tail/stab_func/gauss_process_stab_func/positive/"
short_tail_directory = "results/short_tail/dt_1/300_10/neg_theta/simulations/"
long_tail_directory = "results/long_tail/not_perturbed/"
perturbation_strength = '1_0'

# Get all solution files (from perturbed run)
output_files = []
for path, subdirs, files in os.walk(output_directory):
    if path.split('/')[-1]==perturbation_strength:
        for name in files:
            if 'solution' in name:
                output_files.append(os.path.join(path, name))

path_with_ug = []

for path in output_files:
    path_with_ug.append((path.split('/')[-3], path))

# Get all solution files (from long tail deterministic run)
output_files_long_tail = []
for path, subdirs, files in os.walk(long_tail_directory):
        for name in files:
            if 'solution' in name and 'perturbstr_0.0_sim_0' in name:
                output_files_long_tail.append(os.path.join(path, name))

# Get all solution files (from short tail deterministic run)
output_files_short_tail = []
for path, subdirs, files in os.walk(short_tail_directory):
        for name in files:
            if 'solution' in name and 'perturbstr_0.0' in name:
                output_files_short_tail.append(os.path.join(path, name))


def get_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        t = file["t"][:]
        u = file['u'][:]
        v = file['v'][:]
        z_idx = (np.abs(z - 20)).argmin()
        wind_speed = np.sqrt(u[z_idx, :] ** 2 + v[z_idx, :] ** 2)
        theta = file["theta"][:]
        delta_theta = theta[z_idx, :] - theta[0, :]

    return wind_speed, delta_theta, t, z

uG_range = np.arange(1.0, 2.5, 0.1)

NUM_COLORS = len(uG_range) + 1
cmap = matplotlib.cm.get_cmap("cmc.batlow", NUM_COLORS)
color = cmap.colors
norm = matplotlib.colors.BoundaryNorm(uG_range, cmap.N)

fig, ax = plt.subplots(1, figsize=(5, 5))

# Perturbed run
for tuple in path_with_ug:
    wind_speed, delta_theta, _, _ = get_data(tuple[1])
    uG_idx = np.where(np.isclose(uG_range, float(tuple[0])))[0][0]
    ax.scatter(wind_speed, delta_theta, color=color[uG_idx], s=10, edgecolors=color[uG_idx], alpha=.7)

# Deterministic run with long-tail stability function
mean_long_tail_delta_theta = []
mean_long_tail_wind = []
uG_list = []
for idx, file in enumerate(output_files_long_tail):
    wind_speed, delta_theta, _, _ = get_data(file)
    ax.scatter(wind_speed, delta_theta, color='black', s=10)#, label='long-tail' if idx == 0 else "", marker='s')
    uG_list.append(float(file.split('_')[-5]))
    mean_long_tail_delta_theta.append(np.mean(delta_theta))
    mean_long_tail_wind.append(np.mean(wind_speed))

# Sort by wind speed
_, mean_long_tail_wind, mean_long_tail_delta_theta = zip(*sorted(zip(uG_list, mean_long_tail_wind, mean_long_tail_delta_theta)))

ax.plot(mean_long_tail_wind, mean_long_tail_delta_theta, color='black', label='long-tail')

# Deterministic run with short-tail stability function
mean_short_tail_delta_theta = []
mean_short_tail_wind = []
uG_list = []
for idx, file in enumerate(output_files_short_tail):
    wind_speed, delta_theta, _, _ = get_data(file)
    ax.scatter(wind_speed, delta_theta, color='blue', s=10)#, label='short-tail' if idx == 0 else "")
    uG_list.append(float(file.split('_')[-5]))
    mean_short_tail_delta_theta.append(np.mean(delta_theta))
    mean_short_tail_wind.append(np.mean(wind_speed))

# Sort by wind speed
_, mean_short_tail_wind, mean_short_tail_delta_theta = zip(*sorted(zip(uG_list, mean_short_tail_wind, mean_short_tail_delta_theta)))

ax.plot(mean_short_tail_wind, mean_short_tail_delta_theta, color='blue', label='short-tail', linestyle='dashed')

ax.set_xlabel(r"$s_{20m} [\mathrm{ms^{-1}}]$")
ax.set_ylabel(r"$\Delta \theta$ [K]")
ax.legend(frameon=True, borderpad=0.1, labelspacing=0.1, handletextpad=0.3, handlelength=0.5, loc='upper right', bbox_to_anchor=(1.02, 1.02), markerfirst=False)

ax.set_xlim((0.6, 2.1))
ax.set_ylim((0, 12))

cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
                    label=r"$s_G [\mathrm{ms^{-1}}]$")

plt.savefig(f'{output_directory}stab_func_bifurcation_plot_{perturbation_strength}.png', bbox_inches="tight", dpi=300)
