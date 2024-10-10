import numpy as np
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cmcrameri.cm as cmc

det_data_directory_path = "single_column_model/solution/short_tail_more_uG/simulations/"


def find_solution_files_in_directory(data_path):
    files = []

    for file in os.listdir(data_path):
        if "solution" in file:
            files.append(file)

    # Sort files by uG value
    files = sorted(files, key=lambda x: float(x.split("_")[2].strip()))

    return files


# Get a list of all file names in given directory for u and theta
files = find_solution_files_in_directory(det_data_directory_path)
files = [det_data_directory_path + file for file in files]

sol_dir = {}

for file in files:
    uG = float(file.split("_")[-5])
    with h5py.File(file, "r+") as f:
        time = np.array(f["t"][:]).flatten()
        theta_g = np.array(f["theta_g"][:]).flatten()

    sol_dir[uG] = theta_g

sol_dir["time"] = time

theta_g_all_uG = pd.DataFrame(sol_dir)
theta_g_all_uG["time"] = round(theta_g_all_uG["time"]).astype(int)

hour_idx = np.arange(0, theta_g_all_uG.shape[0], 59)

cooling_rate = {}
for col in theta_g_all_uG.columns:
    cooling_rate[col] = []

for idx, elem in enumerate(hour_idx):
    if idx > 0:
        for col in theta_g_all_uG.columns:
            if col == "time":
                cooling_rate[col].append(theta_g_all_uG[col].iloc[elem])
            else:
                hourly_change = (
                    theta_g_all_uG[col].iloc[elem]
                    - theta_g_all_uG[col].iloc[hour_idx[idx - 1]]
                )
                cooling_rate[col].append(hourly_change)

for key in cooling_rate.keys():
    cooling_rate[key] = np.array(cooling_rate[key]).flatten()

cooling_rate_df = pd.DataFrame(cooling_rate)

cooling_rate_df.set_index("time", inplace=True)


def f(x):
    try:
        return float(x)
    except:
        return x


cooling_rate_df.columns = cooling_rate_df.columns.map(f)

NUM_COLORS = cooling_rate_df.shape[1] + 1
cmap = matplotlib.cm.get_cmap("cmc.batlow", NUM_COLORS)
color = cmap.colors
norm = matplotlib.colors.BoundaryNorm(cooling_rate_df.columns, cmap.N)

fig, ax = plt.subplots(1, figsize=[5, 5])
cooling_rate_df.plot(markevery=1, legend=False, ax=ax, cmap=cmap)
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    orientation="vertical",
    label=r"$s_G \ [\mathrm{ms^{-1}}]$",
)
ax.set_ylabel(r"cooling rate [Kh$^{-1}$]")
ax.set_xlabel("time [h]")
ax.set_xlim(10, 90)
ax.set_ylim(-0.2, 0.1)
plt.savefig(f"{det_data_directory_path}cooling_rate.png", bbox_inches="tight", dpi=300)
