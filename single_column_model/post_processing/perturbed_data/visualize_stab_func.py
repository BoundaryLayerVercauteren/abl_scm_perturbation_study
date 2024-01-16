import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
output_directory = "results/long_tail/stab_func/gauss_process_stab_func/"
perturbation_strength = '0_0'

# Get all solution files
output_files = []
for path, subdirs, files in os.walk(output_directory):
    if path.split('/')[-1]==perturbation_strength:
        for name in files:
            if 'solution' in name:
                output_files.append(os.path.join(path, name))

path_with_ug = []

for path in output_files:
    path_with_ug.append((path.split('/')[-3], path))

def get_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        z_idx = (np.abs(z - 20)).argmin()
        phi = file['phi'][z_idx, :]
        theta = file["theta"][:]
        u = file["u"][:]
        v = file["v"][:]
        richardson =(9.81/300)*((theta[z_idx, :]-theta[0, :])/20)/(((u[z_idx, :]-u[0, :])/20) ** 2 + ((v[z_idx, :]-v[0, :])/20) ** 2)

    return phi, richardson

richardson_dict = {}
phi_dict = {}

for idx, tuple in enumerate(path_with_ug):
    phi, richardson = get_data(tuple[1])
    phi_dict[idx] = phi
    richardson_dict[idx] = richardson

phi_data = pd.DataFrame.from_dict(phi_dict)
richardson_data = pd.DataFrame.from_dict(richardson_dict)

fig, ax = plt.subplots(1, figsize=(10, 10))

for col in phi_data.columns:
    ax.scatter(richardson_data[col], phi_data[col], color='black')

ax.set_xlabel(r"$Ri_{20m}$")
ax.set_ylabel(r"$\phi$")

plt.savefig(f'{output_directory}stab_func_{perturbation_strength}.png', bbox_inches="tight", dpi=300)