import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import scienceplots
import cmcrameri as cram

plt.style.use("science")

# set font sizes for plots
SMALL_SIZE = 18 * 1.5
MEDIUM_SIZE = 22 * 1.5
BIGGER_SIZE = 30 * 1.5

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_directory = "results/long_tail/dt_1/"
perturb_dir = ["neg_theta/", "pos_theta/", "neg_u/", "pos_u/"]
grid_dirs = [
    "100_1/",
    "100_5/",
    "100_10/",
    "200_1/",
    "200_5/",
    "200_10/",
    "300_1/",
    "300_5/",
    "300_10/",
    "400_1/",
    "400_5/",
    "400_10/",
    "500_1/",
    "500_5/",
    "500_10/",
]
sim_directory = "simulations/"


def get_all_data_files(path):
    solution_file_paths = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if "solution" in name:
                solution_file_paths.append(os.path.join(path, name))

    return solution_file_paths


def group_solution_files_by_uG(file_list):
    solution_files_grouped = {}
    u_G_list = []
    r_list = []
    for file in file_list:
        file_name = file.split("/")[-1]
        if float(file_name.split("_")[2]) <= 2.5:
            u_G_list.append(file_name.split("_")[2])
            r_list.append(float(file_name.split("_")[4]))
    u_G_list = np.unique(u_G_list)
    r_list = np.unique(r_list)

    for uG in u_G_list:
        solution_files_grouped[uG] = []

    for file in file_list:
        for uG in u_G_list:
            if f"uG_{uG}" in file:
                solution_files_grouped[uG].append(file)

    return u_G_list, solution_files_grouped, r_list


def calculate_perturbation_strength(variable, cur_r, r_range, height_idx):
    time_idx = 15
    time_variance_data = np.var(variable[:, :time_idx], axis=1)
    # time_variance_data = (variable[:, time_idx] - variable[:, 0]) / (15 * 60)
    # Normalize the variance vector
    normalized_time_variance_data = (time_variance_data - time_variance_data.min()) / (
        time_variance_data.max() - time_variance_data.min()
    ) + 1
    # Normalize the range of perturbation values
    idx_r = np.where(r_range == np.abs(np.round(cur_r, 3)))[0]
    normalized_r_range = (r_range - r_range.min()) / (r_range.max() - r_range.min()) + 1
    # Calculate percentage of perturbation related to variance
    perturbation_percentage = (
        normalized_r_range[idx_r] / normalized_time_variance_data[height_idx] * 100
    )

    return perturbation_percentage[0]


def get_temp_inversion_data(all_file_paths, r_range, z_idx=37):
    df_delta_theta_temp = {}
    perturb_strength_relation = []

    for file_path in all_file_paths:

        try:
            with h5py.File(file_path, "r+") as file:
                t = file["t"][:]
                # Set name of column
                r = file["r"][:][0][0]
                # Calculate temperature inversion
                theta = file["theta"][:]

                if "_theta/" in file_path:
                    perturb_strength = calculate_perturbation_strength(
                        theta, r, r_range, z_idx
                    )
                elif "_u/" in file_path:
                    perturb_strength = calculate_perturbation_strength(
                        file["u"][:], r, r_range, z_idx
                    )

                if (theta == np.nan).any():
                    print(file_path)
                else:
                    df_delta_theta_temp[perturb_strength] = (
                        theta[z_idx, :] - theta[0, :]
                    )
                    perturb_strength_relation.append((r, perturb_strength))

        except Exception as e:
            print(e)
            continue

    df_delta_theta = pd.DataFrame({k: list(v) for k, v in df_delta_theta_temp.items()})
    df_delta_theta = df_delta_theta.reindex(sorted(df_delta_theta.columns), axis=1)
    df_delta_theta["time"] = t.flatten()

    return df_delta_theta, perturb_strength_relation


def calculate_num_sim_with_transition(data):
    columns_with_transitions = []
    for idx, column in enumerate(data.columns):
        if column != "time" and data.iloc[0, idx] >= 5 and (data[column] < 5).any():
            columns_with_transitions.append(column)
        elif column != "time" and data.iloc[0, idx] < 5 and (data[column] > 5).any():
            columns_with_transitions.append(column)

    if len(columns_with_transitions) > 0:
        return np.min(np.abs(columns_with_transitions))
    else:
        return np.nan


def get_transition_statistics(grouped_file_paths, perturb_range):
    min_perturb_strength = {}
    perturbstrength = []
    for key, _ in grouped_file_paths.items():
        data, r_perturb_strength = get_temp_inversion_data(
            grouped_file_paths[key], perturb_range
        )
        min_perturb_strength[key] = calculate_num_sim_with_transition(data)
        if np.isnan(min_perturb_strength[key]):
            perturbstrength.append([key, np.nan, np.nan])
        else:
            idx_r = np.where(r_perturb_strength == min_perturb_strength[key])[0][0]
            perturbstrength.append(
                [key, min_perturb_strength[key], r_perturb_strength[idx_r][0]]
            )
    print(perturbstrength)

    return min_perturb_strength, perturbstrength


labels = [r"$\theta^-$", r"$\theta^+$", r"$u^-$", r"$u^+$"]
markers = ["v", "o", "s", "d"]
colors = matplotlib.cm.get_cmap("cmc.batlow", 5).colors

fig, ax = plt.subplots(5, 3, figsize=(25, 15), sharex=True, sharey=True)
ax = ax.ravel()
for grid_idx, grid_dir in enumerate(grid_dirs):
    for idx, dir in enumerate(perturb_dir):
        directory_path = data_directory + grid_dir + dir + sim_directory

        solution_files = get_all_data_files(directory_path)
        uGs, solution_files_uG, rs = group_solution_files_by_uG(solution_files)
        min_r_for_uG, perturb_translation = get_transition_statistics(
            solution_files_uG, rs
        )
        np.savetxt(
            f"{directory_path}perturb_strength_to_r.txt",
            perturb_translation,
            delimiter=",",
            fmt="%s",
        )

        ax[grid_idx].plot(
            list(min_r_for_uG.keys()),
            list(min_r_for_uG.values()),
            color=colors[idx],
            lw=2,
        )
        ax[grid_idx].scatter(
            min_r_for_uG.keys(),
            min_r_for_uG.values(),
            label=labels[idx],
            marker=markers[idx],
            color=colors[idx],
            s=25,
        )

    if grid_idx == 0 or grid_idx == 1 or grid_idx == 2:
        ax[grid_idx].set_title(rf'$z_s={directory_path.split("/")[3].split("_")[1]}$m')
    if (
        grid_idx == 2
        or grid_idx == 5
        or grid_idx == 8
        or grid_idx == 11
        or grid_idx == 14
    ):
        ax[grid_idx].annotate(
            rf'$t_s={directory_path.split("/")[3].split("_")[0]}$s',
            xy=(1.1, 0.5),
            rotation=90,
            ha="center",
            va="center",
            xycoords="axes fraction",
        )
    if grid_idx == 12 or grid_idx == 13 or grid_idx == 14:
        ax[grid_idx].tick_params(axis="x", rotation=45)
        ax[grid_idx].xaxis.set_major_locator(plt.MaxNLocator(8))
    if grid_idx == 2:
        ax[grid_idx].legend(loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=True)


fig.text(0.5, 0.05, r"$u_G$ [m/s]", ha="center")
fig.text(0.08, 0.5, r"perturbation strength [\%]", va="center", rotation="vertical")


plt.subplots_adjust(wspace=0.02, hspace=0.02)

plt.savefig(
    data_directory + "sensitivity_analysis_variance.png", bbox_inches="tight", dpi=300
)

# Clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close("all")  # Closes all the figure windows.

print(f"Plots for directory: {directory_path} are done!")
