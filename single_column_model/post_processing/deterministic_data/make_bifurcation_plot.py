import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import traceback

import prepare_data, extract_steady_state

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

# Define path to deterministic data
det_directory_path = "single_column_model/solution/long_tail/deterministic/"
det_data_directory_path = det_directory_path + "simulations/"

# Create directory to store visualization
vis_directory_path = os.path.join(det_directory_path, "visualization")
if not os.path.exists(vis_directory_path):
    os.makedirs(vis_directory_path)

# Get a list of all file names in given directory for u and theta
_, _, files_det = prepare_data.find_files_in_directory(det_data_directory_path)

param_range = np.arange(1.0, 2.9, 0.1)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

mean_wind_speed = []
mean_delta_theta = []

NUM_COLORS = len(param_range) + 1
cmap = matplotlib.cm.get_cmap("cmc.batlow", NUM_COLORS)
color = cmap.colors
norm = matplotlib.colors.BoundaryNorm(param_range, cmap.N)

for idx, var in enumerate(param_range):
    try:
        var = np.around(var, 1)

        # Define height at which theta_top is calculated (in meters)
        top_height = 20

        curr_file_det_sim = [s for s in files_det if "_" + str(var) + "_" in s]

        # Make dataframe of deterministic simulation
        (
            df_u_det_sim,
            df_v_det_sim,
            df_delta_theta_det_sim,
            df_tke,
            _,
        ) = prepare_data.create_df_for_fixed_z(det_data_directory_path, curr_file_det_sim, top_height)

        steady_state = extract_steady_state.find_steady_state_fixed_height(df_u_det_sim, df_v_det_sim,
                                                                           df_delta_theta_det_sim, df_tke)

        wind_speed_u = df_u_det_sim.loc[steady_state: steady_state + 60]["sim_0"]
        wind_speed_v = df_v_det_sim.loc[steady_state: steady_state + 60]["sim_0"]
        wind_speed = np.sqrt(wind_speed_u**2+wind_speed_v**2)
        delta_theta = df_delta_theta_det_sim.loc[steady_state: steady_state + 60]["sim_0"]

        mean_wind_speed.append(np.mean(wind_speed))
        mean_delta_theta.append(np.mean(delta_theta))

        if not np.isnan(mean_wind_speed[idx]) and not np.isnan(mean_delta_theta[idx]):
            ax.scatter(
                wind_speed,
                delta_theta,
                label=r"$s_G = $" + str(np.around(var, 1)),
                s=20,
                color=color[idx],
            )

    except Exception:
        mean_wind_speed.append(np.nan)
        mean_delta_theta.append(np.nan)
        print(traceback.format_exc())
        pass

# Add line for mean
ax.plot(
    mean_wind_speed,
    mean_delta_theta,
    label="mean",
    color="grey",
    linewidth=2,
    alpha=0.5,
    linestyle="--",
)

# Add vertical lines to colorbar to indicate transition region
trans_range_uG = []
trans_range_mean_u = []

for idx, val in enumerate(mean_wind_speed):
    for small_idx in range(idx)[1:]:
        if mean_wind_speed[idx] <= mean_wind_speed[small_idx]:
            trans_range_uG.append(param_range[idx])
            trans_range_mean_u.append(mean_wind_speed[idx])
    right_interval = [x for x in range(len(mean_wind_speed)) if x not in range(idx)][1:]
    for larger_idx in right_interval:
        if mean_wind_speed[idx] >= mean_wind_speed[larger_idx]:
            trans_range_uG.append(param_range[idx])
            trans_range_mean_u.append(mean_wind_speed[idx])

# Sort lists
trans_range_mean_u.sort()
trans_range_uG.sort()

try:
    bif_reg = plt.axvspan(
        trans_range_mean_u[0],
        trans_range_mean_u[-1],
        color="r",
        alpha=0.3,
        zorder=0,
        label="bifurcation region",
    )
    plt.legend(handles=[bif_reg])
except Exception:
    print(traceback.format_exc())
    pass

# ax.set_xlim((2, 7))
# ax.set_ylim((0, 12))
ax.set_xlabel("s [m/s]")
ax.set_ylabel(r"$\Delta \theta$ [K]")

cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", label=r"$s_G$ [m/s]")

try:
    cbar.ax.hlines(trans_range_uG[0], 0, 1, colors="r", linewidth=2)
    cbar.ax.hlines(trans_range_uG[-1], 0, 1, colors="r", linewidth=2)
    cbar.ax.vlines(0, trans_range_uG[0], trans_range_uG[-1], colors="r", linewidth=2)
    cbar.ax.vlines(1, trans_range_uG[0], trans_range_uG[-1], colors="r", linewidth=2)
except Exception:
    print(traceback.format_exc())
    pass

plt.savefig(f'{vis_directory_path}/delta_theta_over_u_all_sim_h{top_height}m.png', bbox_inches="tight", dpi=300)