import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import scienceplots
import cmcrameri.cm as cmc
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

# Define directory where simulation output is saved
output_directory = "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/positive/"
perturbation_strength = "0_0"

# Get all solution files
output_files = []
for path, subdirs, files in os.walk(output_directory):
    if path.split("/")[-1] == perturbation_strength:
        for name in files:
            if "solution" in name:
                output_files.append(os.path.join(path, name))

path_with_ug = []

for path in output_files:
    path_with_ug.append((path.split("/")[-3], path))


def get_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        z_idx = (np.abs(z - 40)).argmin()
        phi = file["phi"][:z_idx, :]
        richardson = file['Ri'][:z_idx,:]

    return phi, richardson, z[:z_idx, 0]

data_dict = {}
data_dict['phi'] = []
data_dict['richardson'] = []
data_dict['z'] = []

for idx, tuple in enumerate(path_with_ug):
    phi, richardson, z = get_data(tuple[1])
    phi_array = np.array(phi).flatten('F')
    data_dict['phi'].append(phi_array)
    data_dict['richardson'].append(np.array(richardson).flatten('F'))
    data_dict['z'].append(np.repeat(z, int(len(phi_array)/len(z))))

data_dict['phi'] = np.array(data_dict['phi']).flatten()
data_dict['richardson'] = np.array(data_dict['richardson']).flatten()
data_dict['z'] = np.array(data_dict['z']).flatten()

print(data_dict)
data = pd.DataFrame.from_dict(data_dict)
print(data)
def define_delage_short_tail_stab_function(Ri):
    return 1 + 12 * Ri


def define_delage_long_tail_stab_function(Ri):
    return 1 + 4.7 * Ri


richardson_num = np.linspace(data['richardson'].min(), data['richardson'].max(), 1000)
vec_delage_short_tail_stab_func = np.vectorize(define_delage_short_tail_stab_function)
vec_delage_long_tail_stab_func = np.vectorize(define_delage_long_tail_stab_function)

fig, ax = plt.subplots(1, figsize=(10, 10))

phi_plt = sns.scatterplot(x='richardson',y='phi',data=data,hue='z', ax=ax, palette="cmc.batlow")

ax.colorbar(phi_plt)

ax.plot(
    richardson_num,
    vec_delage_long_tail_stab_func(richardson_num),
    label=r"$long-tail$",
    color='red',
    marker="v",
    markevery=100,
)
ax.plot(
    richardson_num,
    vec_delage_short_tail_stab_func(richardson_num),
    label=r"$short-tail$",
    color='blue',
    marker="s",
    markevery=100,
)

ax.set_xlabel(r"$Ri$")
ax.set_ylabel(r"$\phi$")

plt.savefig(
    f"{output_directory}stab_func_{perturbation_strength}.png",
    bbox_inches="tight",
    dpi=300,
)
