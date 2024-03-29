import cmcrameri.cm as cram
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import traceback
import scienceplots

from single_column_model.post_processing.perturbed_data import prepare_data

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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


def find_files_in_directory(data_path):
    u_files = []
    theta_files = []
    files = []

    for file in os.listdir(data_path):
        if "_u_" in file:
            u_files.append(file)
        elif "_theta_" in file:
            theta_files.append(file)
        elif ".h5" in file:
            files.append(file)

    # Sort files
    u_files = sorted(u_files)  # , key=lambda x: float(x.split("_")[1].strip()))
    theta_files = sorted(theta_files)  # , key=lambda x: float(x.split("_")[1].strip()))
    files = sorted(files)  # , key=lambda x: float(x.split("_")[1].strip()))

    return u_files, theta_files, files


def find_min_max_in_files(directory_path, file_list, variable):
    min_list = []
    max_list = []

    for file in file_list:
        file_path = directory_path + file
        try:
            with h5py.File(file_path, "r+") as cur_file:
                variable_val = cur_file[variable][:]

                min_list.append(np.min(variable_val))
                max_list.append(np.max(variable_val))

        except Exception:
            pass

    return np.min(min_list), np.max(max_list)


def make_3D_plot(
    data_path, vis_path, file_name, curr_param, variable_name, suffix="", z_max=""
):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, "r+") as file:
        if variable_name == "wind_speed":
            u = file["u"][:]
            v = file["v"][:]
            variable_val = np.sqrt(u**2 + v**2)
        elif variable_name == "wind_direction":
            u = file["u"][:]
            v = file["v"][:]
            variable_val = np.arctan2(v, u) * 180 / np.pi
        else:
            variable_val = file[variable_name][:]
        z = file["z"][:]
        t = file["t"][:]

    # Create mesh
    if z_max:
        X, Y = np.meshgrid(t, z[0, :z_max])
    else:
        X, Y = np.meshgrid(t, z)

    # Choose colour
    if variable_name == "theta":
        colours = cram.lajolla
    elif (
        variable_name == "u"
        or variable_name == "wind_direction"
        or variable_name == "wind_speed"
    ):
        colours = cram.davos
    elif variable_name == "v":
        colours = cram.davos

    # Make plot
    plt.figure(figsize=(5, 5))
    if z_max:
        plt.pcolor(X, Y, variable_val[:z_max, :], cmap=colours)
    else:
        plt.pcolor(X, Y, variable_val, cmap=colours)

    if suffix:
        plt.title(r"$u_G = $" + str(curr_param) + ", $r = $" + suffix)
    else:
        plt.title(r"$u_G = $" + str(curr_param))
    plt.xlabel("time [h]")
    plt.ylabel("z [m]")
    plt.xlim((0, 1))
    cbar = plt.colorbar()
    if variable_name == "theta":
        cbar.set_label(r"$\theta$", rotation=0)
    elif variable_name == "wind_direction":
        cbar.set_label(r"$\gamma$", rotation=0)
    elif variable_name == "wind_speed":
        cbar.set_label(r"S", rotation=0)
    else:
        cbar.set_label(variable_name, rotation=0)

    # Save plot
    plt.savefig(
        vis_path
        + "/3D/3D_plots_"
        + variable_name
        + "_"
        + str(curr_param)
        + "_"
        + suffix
        + "_"
        + str(z_max)
        + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def find_z_where_u_const(data_path, file_paths):
    z_idx_dict = {}
    # Open output file and load variables
    for file_idx, file_path in enumerate(file_paths):
        full_file_path = data_path + file_path

        with h5py.File(full_file_path, "r+") as file:
            z = file["z"][:]
            u = file["u"][:]

            # Find max and min for every row
            row_max = np.nanmax(u[:, 10800:], axis=1)
            row_min = np.nanmin(u[:, 10800:], axis=1)
            # Find value range for every row
            row_range = row_max - row_min

            # Find index where z is bigger than 10m and u is near constant
            ten_m_idx = (np.abs(z - 10)).argmin() + 1
            const_u_idx = np.nanargmax(row_range[ten_m_idx:] < 0.3)
            z_idx = const_u_idx + ten_m_idx

            # Set key name
            index_1 = file_path.find("_") + 1
            index_2 = file_path.find("_sim1")
            key_name = file_path[index_1:index_2]
            if z[:, z_idx] > 100:
                z_idx_dict[key_name] = np.nan
            else:
                z_idx_dict[key_name] = z_idx

    # Replace NaN values with mean
    keyList = sorted(z_idx_dict.keys())
    for idx, key in enumerate(z_idx_dict.keys()):
        if np.isnan(z_idx_dict[key]):
            z_idx_dict[key] = z_idx_dict[keyList[idx - 1]]

    return z_idx_dict, z


def plot_delta_theta_over_u(vis_path, data_u, data_delta_theta, suffix):
    data_delta_theta = data_delta_theta.drop(columns=["time"])
    data_u = data_u.drop(columns=["time"])

    plt.figure(figsize=(5, 5))

    plt.scatter(data_u.iloc[:, 0], data_delta_theta.iloc[:, 0], color="black", s=1)

    # data_delta_theta = data_delta_theta.drop(data_delta_theta.index[0:14400])
    # data_u = data_u.drop(data_u.index[0:14400])
    # data_u = data_u.drop(data_u.index[0:14400])

    columns = [float(col) for col in data_delta_theta.columns]

    cmap = matplotlib.cm.get_cmap(
        "cmc.batlow", len(data_delta_theta.columns[:-1])
    ).colors

    for idx, column in enumerate(data_delta_theta.columns[:-1]):
        plt.scatter(data_u[column], data_delta_theta[column], color=cmap[idx], s=1)

    plt.xlabel(r"$u_{top}$ [m/s]")
    plt.ylabel(r"$\Delta \theta$ [K]")

    cbar = plt.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=min(columns), vmax=max(columns)),
            cmap="cmc.batlow",
        )
    )
    cbar.set_label("r", rotation=0)

    plt.savefig(
        vis_path + "/delta_theta_over_u/delta_theta_over_u" + suffix + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def plot_data_over_t(vis_path, data, suffix):
    data = data.set_index("time")

    sm = plt.cm.ScalarMappable(
        cmap="cmc.batlow",
        norm=plt.Normalize(vmin=data.columns.min(), vmax=data.columns.max()),
    )
    sm._A = []

    data.plot(kind="line", colormap="cmc.batlow", legend=False, figsize=(5, 5))

    cbar = plt.colorbar(sm)
    cbar.set_label("r", rotation=0)

    plt.xlabel("time [h]")
    if "delta_theta" in suffix:
        plt.ylabel(r"$\Delta \theta$ [K]")
        # plt.ylim((0, 12))
    elif "u" in suffix:
        plt.ylabel(r"$u$ [m/s]")
        # plt.ylim((0, 4.5))
    elif "v" in suffix:
        plt.ylabel(r"$v$ [m/s]")
        # plt.ylim((0, 1.7))
    else:
        plt.ylabel("TKE [$m^2/s^2$]")
        # plt.ylim((0, 0.2))

    plt.savefig(
        vis_path + "/delta_theta_over_t/var_over_t" + suffix + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def plot_histogram(vis_path, data, variable_name, suffix):
    data = data.drop(columns=["time"])

    plt.figure(figsize=(5, 5))
    # data = data.drop(data.index[0:10800])
    # data.stack().plot.hist(grid=False, bins=10, color='blue')
    # data = data.drop(data.index[0:14400])
    data.stack().plot.hist(grid=False, bins=10, color="black")
    # plt.xlim((0, 10))
    plt.xlabel(r"$\Delta \theta$ [K]")
    plt.title(r"$t \geq 4 h$")

    plt.savefig(
        vis_path + "/histograms/histogram_of_" + variable_name + suffix + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def plot_1D_stoch_process(directory_path, vis_path, file_path):
    full_file_path = directory_path + file_path

    with h5py.File(full_file_path, "r+") as file:
        t = file["t"][:]
        stoch_pro = file["stoch_pro"][:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(t.flatten(), stoch_pro.flatten())
    ax.set_xlabel("t [m/s]")
    ax.set_ylabel("stoch. process")

    plt.savefig(
        vis_path + "/perturbed/stoch_process_" + str(file_path) + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def plot_2D_stoch_process(directory_path, vis_path, file_path):
    full_file_path = directory_path + file_path

    with h5py.File(full_file_path, "r+") as file:
        z = file["z"][:]
        t = file["t"][:]
        stoch_pro = file["perturbation"][:]

    T, Z = np.meshgrid(t, z)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cp = ax.contourf(T, Z, stoch_pro, cmap=cram.lapaz)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 50))

    ax.set_xlabel("t [h]")
    ax.set_ylabel("z [m]")
    fig.colorbar(cp)

    plt.savefig(
        vis_path + "/perturbed/perturbation_" + str(file_path) + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def plot_transitioned_solutions(
    data_path, vis_path, list_files, var, data_delta_theta, z_top
):
    data_delta_theta = data_delta_theta.drop(columns=["time"])

    # # Find all columns where all values are non nan
    # data_delta_theta_subset_non_nan = data_delta_theta[
    #     data_delta_theta.columns[~data_delta_theta.isna().any()]
    # ]
    #
    # # Find all columns where values are increasing
    # def _check_values_increasing(data_col):
    #     values = list(data_col)
    #     return all(values[idx] <= values[idx + 1] for idx in range(len(values) - 1))
    #
    # data_delta_theta_subset = pd.DataFrame()
    # for column in data_delta_theta_subset_non_nan.columns:
    #     col_subset = (
    #         data_delta_theta_subset_non_nan[column]
    #         .rolling(2)
    #         .agg(_check_values_increasing)
    #     )
    #     if (col_subset.iloc[-10:] == 0).any():
    #         continue
    #     else:
    #         data_delta_theta_subset[column] = data_delta_theta_subset_non_nan[column]

    # Find corresponding file
    for column in data_delta_theta.columns:
        file_name = list_files.iloc[0, data_delta_theta.columns.get_loc(column)]
        make_3D_plot(
            data_path, vis_path, file_name, var, "u", suffix=str(np.round(column, 3))
        )
        make_3D_plot(
            data_path, vis_path, file_name, var, "v", suffix=str(np.round(column, 3))
        )
        make_3D_plot(
            data_path,
            vis_path,
            file_name,
            var,
            "theta",
            suffix=str(np.round(column, 3)),
        )
        make_3D_plot(
            data_path,
            vis_path,
            file_name,
            var,
            "wind_speed",
            suffix=str(np.round(column, 3)),
        )
        make_3D_plot(
            data_path,
            vis_path,
            file_name,
            var,
            "wind_direction",
            suffix=str(np.round(column, 3)),
        )

        # make_3D_plot(data_path, vis_path, file_name, var, "u", suffix=str(column), z_max=z_top)
        # make_3D_plot(data_path, vis_path, file_name, var, "v", suffix=str(column), z_max=z_top)
        # make_3D_plot(data_path, vis_path, file_name, var, "theta", suffix=str(column), z_max=z_top)


if __name__ == "__main__":
    # Define path to stochastic data
    data_directory_path = "results/long_tail/dt_1/300_10/neg_u/"
    data_directory_path_single = data_directory_path + "simulations/"

    # Create directory to store visualization
    vis_directory_path = os.path.join(data_directory_path, "visualization")
    if not os.path.exists(vis_directory_path):
        os.makedirs(vis_directory_path)
        os.makedirs(vis_directory_path + "/perturbed")
        os.makedirs(vis_directory_path + "/delta_theta_over_t")
        os.makedirs(vis_directory_path + "/delta_theta_over_u")
        os.makedirs(vis_directory_path + "/histograms")
        os.makedirs(vis_directory_path + "/3D")

    # Get a list of all file names in given directory for u and theta
    _, _, files_sin = find_files_in_directory(data_directory_path_single)

    for var in np.array([2.3]):  # np.arange(1.8, 1.8, 0.1):
        try:
            var = np.around(var, 1)

            # Define height at which theta_top is calculated
            top_height = 20
            idx_bl_top_height = 37
            # Get all files which correspond to current loop
            curr_files_single_sim = [s for s in files_sin if "_" + str(var) + "_" in s]

            # Make dataframe of all single simulations
            (
                df_u_sing_sim,
                df_v_sing_sim,
                df_delta_theta_sing_sim,
                df_tke,
                df_files_names,
            ) = prepare_data.create_df_for_fixed_z(
                data_directory_path_single,
                curr_files_single_sim,
                top_height,
                "stochastic",
            )

            # Make 3D plot of time series with transitions
            plot_transitioned_solutions(
                data_directory_path_single,
                vis_directory_path,
                df_files_names,
                var,
                df_delta_theta_sing_sim,
                idx_bl_top_height,
            )

            # # Make histogram for delta theta (i.e. theta_top - theta_0) (single simulations)
            # plot_histogram(vis_directory_path, df_delta_theta_sing_sim, 'delta_theta', '_' + str(var))

            # # Plot delta theta over u (single simulations)
            # plot_delta_theta_over_u(vis_directory_path, df_u_sing_sim, df_delta_theta_sing_sim, '_' + str(var))

            # Plot delta theta over t (single simulations)
            plot_data_over_t(
                vis_directory_path, df_delta_theta_sing_sim, "_delta_theta_" + str(var)
            )
            #
            # # Plot u over t (single simulations)
            # plot_data_over_t(vis_directory_path, df_u_sing_sim, '_u_' + str(var))
            #
            # # Plot v over t (single simulations)
            # plot_data_over_t(vis_directory_path, df_v_sing_sim, '_v_' + str(var))
            #
            # # Plot TKE over t (single simulations)
            # plot_data_over_t(vis_directory_path, df_tke, '_tke_' + str(var))

            # # Plot one random stochastic process
            # for file in curr_files_single_sim:
            #     if file=='solution_uG_1.7_perturbstr_0.01_sim_0.h5':
            #         random_file = file
            #         #random_file = random.choice(curr_files_single_sim)
            #         plot_2D_stoch_process(data_directory_path_single, vis_directory_path, random_file)

        except Exception:
            print(traceback.format_exc())
            pass
