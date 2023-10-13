"""Needs to be run with: python3 -m single_column_model.post_processing.extract_steady_state"""
import os
import traceback
import warnings
from functools import reduce

warnings.simplefilter(action='ignore', category=FutureWarning)

import cmcrameri.cm as cram
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from single_column_model.post_processing import (
    prepare_data, visualize_deterministic_model_output)


def find_steady_state_fixed_height(data_u, data_v, data_delta_theta, data_tke):
    """A steady state is defined as a time point where the value minus a 3 hours average is smaller than 0.1."""

    # Calculate rolling mean
    num_hours = 5
    one_h_num_steps = data_u.index[np.round(data_u["time"], 3) == 1.0].tolist()[0]

    data_u["rol_mean"] = (
        data_u["sim_0"].rolling(num_hours * one_h_num_steps, min_periods=1).mean()
    )
    data_v["rol_mean"] = (
        data_v["sim_0"].rolling(num_hours * one_h_num_steps, min_periods=1).mean()
    )
    data_delta_theta["rol_mean"] = (
        data_delta_theta["sim_0"].rolling(num_hours * one_h_num_steps, min_periods=1).mean()
    )
    data_tke["rol_mean"] = (
        data_tke["sim_0"].rolling(num_hours * one_h_num_steps, min_periods=1).mean()
    )

    # Drop first 8 hours
    eighth_h_index = 8 * one_h_num_steps

    data_u = data_u.iloc[eighth_h_index:, :].copy()
    data_v = data_v.iloc[eighth_h_index:, :].copy()
    data_delta_theta = data_delta_theta.iloc[eighth_h_index:, :].copy()
    data_tke = data_tke.iloc[eighth_h_index:, :].copy()

    # Calculate difference to mean for each time point
    data_u["diff_mean"] = np.abs(data_u["rol_mean"] - data_u["sim_0"])
    data_v["diff_mean"] = np.abs(data_v["rol_mean"] - data_v["sim_0"])
    data_delta_theta["diff_mean"] = np.abs(
        data_delta_theta["rol_mean"] - data_delta_theta["sim_0"]
    )
    data_tke["diff_mean"] = np.abs(data_tke["rol_mean"] - data_tke["sim_0"])

    # Find all values where the difference is less or equal to 2% of the maximum of the corresponding value
    deviation_percentage = 0.03
    data_u["bel_thresh"] = data_u["diff_mean"] <= deviation_percentage
    data_v["bel_thresh"] = data_v["diff_mean"] <= deviation_percentage
    data_delta_theta["bel_thresh"] = data_delta_theta["diff_mean"] <= deviation_percentage
    data_tke["bel_thresh"] = data_tke["diff_mean"] <= deviation_percentage

    # Find continuous time series where the deviance from the mean is below the threshold for at least one hour
    steady_state = np.nan

    steady_state_range_u = []
    steady_state_range_v = []
    steady_state_range_delta_theta = []
    steady_state_range_tke = []

    len_steady_state_h = 1
    for k, v in data_u.groupby(
            (data_u["bel_thresh"].shift() != data_u["bel_thresh"]).cumsum()
    ):
        if v["bel_thresh"].all():
            if len(v) >= len_steady_state_h * one_h_num_steps:
                steady_state_range_u = np.append(steady_state_range_u, v.index.tolist())

    for k, v in data_v.groupby(
            (data_v["bel_thresh"].shift() != data_v["bel_thresh"]).cumsum()
    ):
        if v["bel_thresh"].all():
            if len(v) >= len_steady_state_h * one_h_num_steps:
                steady_state_range_v = np.append(steady_state_range_v, v.index.tolist())

    for k, v in data_delta_theta.groupby(
            (
                    data_delta_theta["bel_thresh"].shift() != data_delta_theta["bel_thresh"]
            ).cumsum()
    ):
        if v["bel_thresh"].all():
            if len(v) >= len_steady_state_h * one_h_num_steps:
                steady_state_range_delta_theta = np.append(
                    steady_state_range_delta_theta, v.index.tolist()
                )

    for k, v in data_tke.groupby(
            (data_tke["bel_thresh"].shift() != data_tke["bel_thresh"]).cumsum()
    ):
        if v["bel_thresh"].all():
            if len(v) >= len_steady_state_h * one_h_num_steps:
                steady_state_range_tke = np.append(
                    steady_state_range_tke, v.index.tolist()
                )

    try:
        steady_state = reduce(
            np.intersect1d,
            (
                steady_state_range_u,
                steady_state_range_v,
                steady_state_range_delta_theta,
                steady_state_range_tke,
            ),
        )[0]
    except Exception:
        pass

    return steady_state


def plot_time_differences(data_path, vis_path, file_name, curr_param, variable_name):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, "r+") as file:
        # perform byteswap to make handling with pandas dataframe possible
        variable_val = file[variable_name][:]#.byteswap().newbyteorder()
        z = file["z"][:]#.byteswap().newbyteorder()
        t = file["t"][:]#.byteswap().newbyteorder()

    data = pd.DataFrame(data=variable_val.T, columns=z.flatten())
    data["time"] = t.flatten()

    # Calculate rolling mean
    one_h_num_steps_idx = data.index[np.round(data["time"], 3) == 1.0].tolist()[0]
    data_rol_mean = data.rolling(3 * one_h_num_steps_idx, min_periods=1).mean()

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Make plot
    plt.figure(figsize=(5, 5))

    data = data.drop(columns=["time"])
    data_rol_mean = data_rol_mean.drop(columns=["time"])
    plt.pcolor(
        X, Y, np.abs(data - data_rol_mean).T, cmap=cram.bilbao
    )  # , vmin=0.0, vmax=1.0)
    plt.axhline(y=z[29, :], color="black", linestyle="--")

    # plt.title(r'$u_G = $' + str(curr_param))
    plt.xlabel("time [h]")
    plt.ylabel("z [m]")
    cbar = plt.colorbar()
    if variable_name == "theta":
        cbar.set_label(r"$\tilde{\theta}$", rotation=0)
    elif variable_name == "u":
        cbar.set_label(r"$\tilde{u}$", rotation=0)
    elif variable_name == "v":
        cbar.set_label(r"$\tilde{v}$", rotation=0)
    elif variable_name == "TKE":
        cbar.set_label(r"$\tilde{TKE}$", rotation=0)

    # Save plot
    plt.savefig(
        vis_path
        + "/3D_plots_increments_"
        + variable_name
        + "_"
        + str(curr_param)
        + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def plot_inversion_strength(
        data_path, vis_path, file_name, curr_param, steady_state_coord
):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, "r+") as file:
        # perform byteswap to make handling with pandas dataframe possible
        theta = file["theta"][:]#.byteswap().newbyteorder()
        z = file["z"][:]
        t = file["t"][:]

    theta_df = pd.DataFrame(data=theta.T, columns=z.flatten())

    # Calculate inversion strength
    delta_theta_df = theta_df.copy()
    for col in theta_df.columns:
        delta_theta_df[col] = theta_df[col] - theta_df.iloc[:, 0]

    # # Calculate height of inversion
    # inversion_height_idx = np.zeros((1, len(t.flatten())))
    # # inversion_height_idx[...] = np.nan
    # for row_idx in delta_theta_df.index[1:]:
    #     inversion_height_idx[0, row_idx] = np.argmax(
    #         np.around(delta_theta_df.iloc[row_idx, :], 0) == 0.0
    #     )

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Make plot
    plt.figure(figsize=(5, 5))

    plt.pcolor(X, Y, delta_theta_df.T, cmap=cram.bilbao_r)
    # plt.plot(
    #     t.flatten(),
    #     z[list(map(int, inversion_height_idx.flatten())), :],
    #     color="black",
    #     linestyle="--",
    #     label="inversion height",
    # )

    # plt.ylim((1,100))
    plt.xlabel("time [h]")
    plt.ylabel("z [m]")
    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta\theta$", rotation=0)

    if steady_state_coord:
        plt.scatter(
            t.flatten()[int(steady_state_coord[0])],
            steady_state_coord[1],
            c="black",
            s=8,
            label="steady state",
            marker="x",
        )

    leg = plt.legend()
    for text in leg.get_texts():
        text.set_color("white")
    # Save plot
    plt.savefig(
        vis_path
        + "/inversion_strength_"
        + str(curr_param)
        + "_h"
        + str(int(np.around(steady_state_coord[1])))
        + ".png",
        bbox_inches="tight",
        dpi=300,
    )

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def find_Ekman_layer_height(
        data_path, vis_path, file_name, u_G, steady_state_coord=None, make_plot=True
):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, "r+") as file:
        # perform byteswap to make handling with pandas dataframe possible
        u = file["u"][:]#.byteswap().newbyteorder()
        z = file["z"][:]  # #.byteswap().newbyteorder()
        t = file["t"][:]  # #.byteswap().newbyteorder()

    data = pd.DataFrame(data=u.T, columns=z.flatten())

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for row_idx in data.index:
        ekman_height_idx[0, row_idx] = np.argmax(
            np.around(data.iloc[row_idx, :], 1) == u_G
        )

    if make_plot:
        # Create mesh
        X, Y = np.meshgrid(t, z)

        # Make plot
        plt.figure(figsize=(5, 5))
        plt.pcolor(X, Y, u, cmap=cram.davos_r)
        plt.plot(
            t.flatten(),
            z[list(map(int, ekman_height_idx.flatten())), :],
            color="red",
            linestyle="--",
            label="Ekman layer",
        )

        cbar = plt.colorbar()
        cbar.set_label("u", rotation=0, labelpad=2)

        if steady_state_coord:
            plt.scatter(
                t.flatten()[int(steady_state_coord[0])],
                steady_state_coord[1],
                c="black",
                s=8,
                label="steady state",
                marker="x",
            )
        plt.ylim((0, 50))
        # plt.title(r"$u_G = $" + str(u_G))
        plt.xlabel("time [h]")
        plt.ylabel("z [m]")
        leg = plt.legend()
        for text in leg.get_texts():
            text.set_color("white")

        # Save plot
        if steady_state_coord:
            plt.savefig(
                vis_path
                + "/Ekman_layer_"
                + str(u_G)
                + "_h"
                + str(int(np.around(steady_state_coord[1])))
                + ".png",
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.savefig(
                vis_path + "/Ekman_layer_" + str(u_G) + ".png",
                bbox_inches="tight",
                dpi=300,
            )


        # Clear memory
        plt.cla()  # Clear the current axes.
        plt.clf()  # Clear the current figure.
        plt.close("all")  # Closes all the figure windows.

    return z[list(map(int, ekman_height_idx.flatten())), :].flatten()


def calculate_stable_BL_height_based_on_Richardson_number(data_path, vis_path, file_name, u_G, steady_state_coord=None):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, "r+") as file:
        # perform byteswap to make handling with pandas dataframe possible
        u = file["u"][:]#.byteswap().newbyteorder()
        v = file["v"][:]#.byteswap().newbyteorder()
        theta = file["theta"][:]#.byteswap().newbyteorder()
        z = file["z"][:]
        t = file["t"][:]
        Ri = file["Ri"][:]#.byteswap().newbyteorder()

    g = 9.81
    theta_ref = 290
    z_0 = 0.044

    #Ri = (g / theta_ref) * (z - z_0) * (theta - theta[0, :]) / ((u - u[0, :]) ** 2 + (v - v[0, :]) ** 2)

    Ri_critical = 0.5
    idx_Ri_eq_cr = []
    for time_idx in np.arange(0, np.shape(Ri)[1]):
        idx_Ri_eq_cr.append(np.nanargmax((Ri[:, time_idx] > Ri_critical)[1:]) + 1)

    X, Y = np.meshgrid(t, z)

    plt.figure(figsize=(5, 5))
    plt.pcolor(X, Y, Ri, cmap=cram.davos_r)
    plt.plot(
        t.flatten(),
        z[idx_Ri_eq_cr, :],
        color="red",
        linestyle="--",
        label="critical Richardson number",
    )

    plt.ylim((0,50))

    cbar = plt.colorbar()
    cbar.set_label("Ri", rotation=0, labelpad=2)

    plt.xlabel("time [h]")
    plt.ylabel("z [m]")

    plt.legend()

    plt.savefig(f'{vis_path}/Richardson_number_{u_G}.png', bbox_inches="tight", dpi=300)

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


def extract_initial_cond(
        curr_steady_state, data_file_path, init_file_path, variable_name
):
    with h5py.File(data_file_path, "r+") as file:
        variable_val = file[variable_name][:]
        time = file['t'][:]

    idx = 850
    initial_cond = variable_val[:, int(curr_steady_state)]
    #print(time[:,idx])
    np.save(init_file_path + variable_name, initial_cond)


if __name__ == "__main__":
    # Define path to deterministic data
    stab_func_type = 'long_tail'
    det_directory_path = f"single_column_model/solution/{stab_func_type}/deterministic/"
    det_data_directory_path = det_directory_path + "simulations/"

    # Create directory to store visualization
    vis_directory_path = os.path.join(det_directory_path, "visualization")
    if not os.path.exists(vis_directory_path):
        os.makedirs(vis_directory_path)

    # Get a list of all file names in given directory for u and theta
    _, _, files_det = prepare_data.find_files_in_directory(det_data_directory_path)

    # bl_top_height_det_sim_dict, z = prepare_data.find_z_where_u_const(det_data_directory_path, files_det)

    for var in np.arange(5.0, 5.1, 0.1):
        try:
            var = np.around(var, 1)

            curr_file_det_sim = [s for s in files_det if "_" + str(var) + "_" in s]

            # Plot time differences
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'theta')
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'u')
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'v')
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'TKE')

            # Make dataframe of simulation
            bl_top_height_det_sim = (
                20  # idx = 37#z[height_idx,:] #bl_top_height_det_sim_dict[str(var)], :]
            )
            bl_top_height_det_sim_idx = 37

            df_u, df_v, df_delta_theta, df_tke, _ = prepare_data.create_df_for_fixed_z(
                det_data_directory_path, curr_file_det_sim, bl_top_height_det_sim
            )

            # calculate_stable_BL_height_based_on_Richardson_number(det_data_directory_path, vis_directory_path,
            #                                                       curr_file_det_sim[0], var)

            steady_state = find_steady_state_fixed_height(
                df_u, df_v, df_delta_theta, df_tke
            )
            print(var, steady_state)
            # #Plot variables over time at BL height
            # visualize_deterministic_model_output.plot_combined_data_over_t(vis_directory_path, df_u, df_v,
            #                                                                df_delta_theta, df_tke,
            #                                                                f'_z_20_steady_{var}',
            #                                                                steady_state)

            # Extract initial condition
            init_dir_path = "single_column_model/init_condition/"
            init_file_path = (
                    init_dir_path + f"{stab_func_type}_steady_state_Ug" + str(var) + "_"
            )
            extract_initial_cond(
                steady_state,
                det_data_directory_path + curr_file_det_sim[0],
                init_file_path,
                "theta",
            )
            extract_initial_cond(
                steady_state,
                det_data_directory_path + curr_file_det_sim[0],
                init_file_path,
                "u",
            )
            extract_initial_cond(
                steady_state,
                det_data_directory_path + curr_file_det_sim[0],
                init_file_path,
                "v",
            )
            extract_initial_cond(
                steady_state,
                det_data_directory_path + curr_file_det_sim[0],
                init_file_path,
                "TKE",
            )
            # continue
            # Plot inversion strength
            # plot_inversion_strength(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var,
            #                         [steady_state, bl_top_height_det_sim])
            #
            # # Plot Ekman layer height
            # _ = find_Ekman_layer_height(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var,
            #                             [steady_state, bl_top_height_det_sim])

        except Exception:
            print(traceback.format_exc())
            pass
