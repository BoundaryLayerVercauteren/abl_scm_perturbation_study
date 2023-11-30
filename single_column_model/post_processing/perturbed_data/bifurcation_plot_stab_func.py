import os
from itertools import groupby
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

# Define directory where simulation output is saved
output_directory = "results/long_tail/stab_func/gauss_process_stab_func/"
perturbation_strength = '0_0'

# Get all solution files
output_files = []
for path, subdirs, files in os.walk(output_directory):
    if path.split('/')[-1]==perturbation_strength:
        for name in files:
            if 'solution' in name:
                output_files.append(os.path.join(path, name))

print(output_files)
# Group them by directory, i.e. uG
grouped_output_files = [list(g) for _, g in groupby(output_files.sort(), lambda k: k[-3])]
print(grouped_output_files)

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
norm = matplotlib.colors.BoundaryNorm(param_range, cmap.N)

fig, ax = plt.subplots(1, figsize=(5, 10))

for group in grouped_output_files:
    for file in group:
        wind_speed, delta_theta, _, _ = get_data(file)
        uG = file.split('/')[-3]
        uG_idx = np.where(uG_range==float(uG))[0][0]
        ax.scatter(wind_speed, delta_theta, s=20, color=color[uG_idx])

ax.set_xlabel(r"s_{20m} [m/s]")
ax.set_ylabel(r"$\Delta \theta$ [K]")
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
                    label=r"$s_G [\mathrm{ms^{-1}}]$")

plt.savefig(f'{output_directory}stab_func_bifurcation_plot.png', bbox_inches="tight", dpi=300)
