import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Define directory where simulation output is saved
output_directory = "results/long_tail/stab_func/gauss_process_stab_func/positive/"
# ---------------------------------------------------------------------------
# Define where the unstable equilibrium is located
location_unstable_eq = 5
# ---------------------------------------------------------------------------
def check_if_transitioned(values):
    if values[0] > location_unstable_eq:
        if any(values < location_unstable_eq):
            return 1
        else:
            return 0
    else:
        if any(values > location_unstable_eq):
            return 1
        else:
            return 0


def combine_info_about_simulation_type(file_path):
    try:
        with h5py.File(file_path, "r+") as file:
            z = file["z"][:]
            z_idx = (np.abs(z - 20)).argmin()
            theta = file["theta"][:]
            delta_theta = theta[z_idx, :] - theta[0, :]
    except OSError:
        return

    # Extract u value for simulation from file path
    file_name = np.array(file_path.split("/"))[-1]
    file_elements = np.array(file_name.split("_"))

    u_val = float(file_elements[2])
    sigma_s = float(file_elements[4])
    sim_idx = int(float(file_elements[-1][:-3]))

    # Calculate how many transitions take place on average over all simulations
    transitioned = check_if_transitioned(delta_theta.flatten())

    # Save parameter combination for this simulation
    return (u_val, sigma_s, sim_idx, transitioned)


# Get all solution files
output_files = []
for path, subdirs, files in os.walk(output_directory):
    for name in files:
        output_files.append(os.path.join(path, name))

parameter_comb = []

with ProcessPoolExecutor(max_workers=125) as executor:
    for result in executor.map(combine_info_about_simulation_type, output_files):
        if result:
            parameter_comb.append(result)

# Save calculated values in file
with open(f"{output_directory}transition_overview.json", "w") as file:
    json.dump(parameter_comb, file)
