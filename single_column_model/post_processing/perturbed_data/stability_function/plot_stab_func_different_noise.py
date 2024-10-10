"""Script for figure 10"""

import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(10, 10))

# Define directory where simulation output is saved
output_directories = [
    "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/positive/1.0/1_0/",
    "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/positive/1.0/0_0/",
    "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/negative/1.0/0_07/",
    "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/negative/1.0/1_0/",
]

simulation_indices = np.arange(0, 100, 1)


def get_solution_files(path, sim_range):
    output_files = []
    for root, _, files in os.walk(path):
        for name in files:
            if "solution" in name and any(
                f"sim_{sim_idx}.0" in name for sim_idx in sim_range
            ):
                output_files.append(os.path.join(root, name))
    return output_files


def get_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        phi = file["phi"][:]
        t = file["t"][:].flatten()
        z = file["z"][:]

    z_idx = (np.abs(z - 20)).argmin()

    return phi[z_idx, :], t


titles = ["a)", "b)", "c)", "d)"]
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, constrained_layout=True)
ax = ax.ravel()

for idx, dir in enumerate(output_directories):
    solution_files = get_solution_files(dir, simulation_indices)

    phi_all_sim = []
    for file in solution_files:
        phi_data, time = get_data(file)
        phi_all_sim.append(phi_data)

    phi_all_sim = np.array(phi_all_sim)
    phi_all_sim = pd.DataFrame(data=phi_all_sim.T, index=time)

    phi_all_sim.plot(ax=ax[idx], colormap="cmc.batlow_r", legend=False)

    ax[idx].set_title(titles[idx], loc="left")
    if idx % 2 == 0:
        ax[idx].set_ylabel(r"$\rho_{20m}$")
    if idx > 1:
        ax[idx].set_xlabel("t [h]")
    ax[idx].set_xlim((0, 1))


plt.savefig(
    "single_column_model/solution/short_tail/perturbed/stab_func/visualizations/stab_func_multi_noise.png",
    bbox_inches="tight",
    dpi=300,
)
