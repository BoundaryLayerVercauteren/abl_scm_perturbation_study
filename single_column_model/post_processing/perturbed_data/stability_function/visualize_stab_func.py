import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import scienceplots
import cmcrameri.cm as cmc
import seaborn as sns
import traceback


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
output_directory = "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/negative/"
perturbation_strength = "-1.0"

# Get all solution files
output_files = []
for path, subdirs, files in os.walk(output_directory):
    for name in files:
        if "solution" in name:
            output_files.append(os.path.join(path, name))


def get_data(full_file_path):
    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        # z_idx = (np.abs(z - 40)).argmin()
        phi = file["phi"][:].flatten()  # [:z_idx, :]
        richardson = file["Ri"][:].flatten()  #:z_idx,:]

    return phi, richardson, np.tile(z, int(len(phi) / len(z))).flatten()


# data_dict = {}
# data_dict['phi'] = np.array([])
# data_dict['richardson'] = np.array([])
# data_dict['z'] = np.array([])
#
# for idx, file in enumerate(output_files):
#     print(idx, len(output_files))
#     try:
#         phi, richardson, z = get_data(file)
#         data_dict['phi']= np.append(data_dict['phi'], phi)
#         data_dict['richardson']= np.append(data_dict['richardson'], richardson)
#         data_dict['z']= np.append(data_dict['z'], z)
#     except Exception:
#         print(traceback.format_exc())
#         pass
#     # Store data in between to free up memory
#     if idx>0 and (idx%100==0 or idx==len(output_files)):
#         data_dict['phi'] = np.array(data_dict['phi']).flatten()
#         data_dict['richardson'] = np.array(data_dict['richardson']).flatten()
#         data_dict['z'] = np.array(data_dict['z']).flatten()
#
#         data = pd.DataFrame.from_dict(data_dict)
#         data.to_csv(f'/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/phi_summary_{perturbation_strength}/summary_{idx}.csv', index=False)
#
#         del data
#         data_dict['phi'] = np.array([])
#         data_dict['richardson'] = np.array([])
#         data_dict['z'] = np.array([])

path = f"/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/phi_summary_{perturbation_strength}/"
file_list = os.listdir(path)
file_path_list = [os.path.join(path, file) for file in file_list if ".png" not in file]

data = pd.DataFrame(pd.read_csv(file_path_list[0]))

for file_path in file_path_list:
    sub_data = pd.read_csv(file_path)
    data = pd.concat([data, sub_data], ignore_index=True)

# Reduce size of data frame
data = data.drop(data[data["z"] > 50].index)
data["z"] = data["z"].astype("int")
data = data.round(3)
data.drop_duplicates(inplace=True)


def define_delage_short_tail_stab_function(Ri):
    return 1 + 12 * Ri


def define_delage_long_tail_stab_function(Ri):
    return 1 + 4.7 * Ri


richardson_num = np.linspace(data["richardson"].min(), data["richardson"].max(), 1000)
vec_delage_short_tail_stab_func = np.vectorize(define_delage_short_tail_stab_function)
vec_delage_long_tail_stab_func = np.vectorize(define_delage_long_tail_stab_function)

fig, ax = plt.subplots(1, figsize=(10, 10))

norm = plt.Normalize(data["z"].min(), data["z"].max())
sm = plt.cm.ScalarMappable(cmap="cmc.batlow", norm=norm)
sns.scatterplot(
    x="richardson",
    y="phi",
    data=data,
    ax=ax,
    legend=False,
    s=2,
    linewidth=0,
    hue="z",
    palette="cmc.batlow",
    alpha=0.7
)
ax.figure.colorbar(sm, ax=ax, label="z [m]")

ax.plot(
    richardson_num,
    vec_delage_long_tail_stab_func(richardson_num),
    label="long-tail",
    color="red",
    marker="v",
    markevery=10,
)
ax.plot(
    richardson_num,
    vec_delage_short_tail_stab_func(richardson_num),
    label="short-tail",
    color="blue",
    marker="s",
    markevery=10,
)

ax.legend(loc="upper left")

ax.set_xscale("log")
ax.set_xlabel(r"$Ri$")
ax.set_ylabel(r"stability function")

ax.set_ylim(0, 10)

plt.savefig(
    f"/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/stab_func/gauss_process_stab_func/phi_summary_{perturbation_strength}/stab_func_{perturbation_strength}.png",
    bbox_inches="tight",
    dpi=300,
)
