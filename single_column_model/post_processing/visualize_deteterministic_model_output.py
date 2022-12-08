import os
import traceback

import cmcrameri.cm as cram
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from single_column_model.post_processing import prepare_data, extract_steady_state

matplotlib.use('webagg')
plt.style.use('science')

# set font sizes for plots
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def make_3D_plot(data_path, vis_path, file_name, curr_param, variable_name):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, 'r+') as file:
        variable_val = file[variable_name][:]
        z = file['z'][:]
        t = file['t'][:]

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Choose colour
    if variable_name == 'theta':
        colours = cram.lajolla
        v_min = 289
        v_max = 302
    elif variable_name == 'u':
        colours = cram.davos
        v_min = -2.5
        v_max = 7.5
    elif variable_name == 'v':
        colours = cram.davos
        v_min = 0.0
        v_max = 3.5
    elif variable_name == 'TKE':
        colours = cram.tokyo

    # Make plot
    plt.figure(figsize=(5, 5))
    plt.pcolor(X, Y, variable_val, cmap=colours)  # vmin=v_min, vmax=v_max)
    plt.title(r'$u_G = $' + str(curr_param))
    # plt.ylim((0,100))
    plt.xlabel('time [h]')
    plt.ylabel('z [m]')
    cbar = plt.colorbar()
    if variable_name == 'theta':
        cbar.set_label(r'$\theta$', rotation=0, labelpad=2)
    else:
        cbar.set_label(variable_name, rotation=0, labelpad=2)

    # Save plot
    plt.savefig(vis_path + '/3D_plots_' + variable_name + '_' + str(curr_param) + '.png', bbox_inches='tight', dpi=300)

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def make_3D_plot_increments(data_path, vis_path, file_name, curr_param, variable_name):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, 'r+') as file:
        variable_val = file[variable_name][:]
        z = file['z'][:]
        t = file['t'][:]

    # Calculate increments
    moving_sum = np.zeros(np.shape(variable_val))
    moving_sum[...] = np.nan

    for col_idx in np.arange(1800, np.shape(variable_val)[1]):
        moving_sum[:, col_idx] = np.sum(variable_val[:, col_idx - 1800:col_idx], axis=1) / 1800

    increments = np.abs(variable_val - moving_sum)

    # Calculate sum of increments for all z
    min_idx_increment_small = False
    sum_increments = np.max(increments, axis=0)
    # Find indices where the mean is very small
    idx_small_increments = np.argwhere(sum_increments < 0.1).flatten()
    if len(idx_small_increments) != 0:
        # Check if values indices are consecutive
        diff_idx_small_increments = np.diff(idx_small_increments)
        last_idx_non_consecutive = np.argwhere(diff_idx_small_increments != 1)
        if len(last_idx_non_consecutive) == 0:
            min_idx_increment_small = idx_small_increments[0]
        else:
            min_idx_increment_small = idx_small_increments[last_idx_non_consecutive.max() + 1]

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Make plot
    plt.figure(figsize=(5, 5))

    plt.pcolor(X, Y, increments, cmap=cram.bilbao)  # , vmin=0.0, vmax=1.0)
    plt.axhline(y=z[29, :], color='black', linestyle='--')

    if min_idx_increment_small:
        plt.axvline(x=t[:, min_idx_increment_small], color='red', linestyle='--')

    # plt.title(r'$u_G = $' + str(curr_param))
    plt.xlabel('time [h]')
    plt.ylabel('z [m]')
    cbar = plt.colorbar()
    if variable_name == 'theta':
        cbar.set_label(r'$\tilde{\theta}$', rotation=0)
    elif variable_name == 'u':
        cbar.set_label(r'$\tilde{u}$', rotation=0)
    elif variable_name == 'v':
        cbar.set_label(r'$\tilde{v}$', rotation=0)

    # Save plot
    plt.savefig(vis_path + '/3D_plots_increments_' + variable_name + '_' + str(curr_param) + '.png',
                bbox_inches='tight', dpi=300)

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def plot_data_over_t(vis_path, data, suffix, steady_stat):
    plt.figure(figsize=(15, 5))

    plt.plot(data['time'], data['sim1'], color='black')
    if not np.isnan(steady_stat):
        plt.scatter(data.loc[steady_stat, 'time'], data.loc[steady_stat, 'sim1'], color='red', s=30)

    plt.xlabel('time [h]')
    if 'delta_theta' in suffix:
        plt.ylabel(r'$\Delta \theta$ [K]')
        plt.ylim((0, 12))
    elif 'u_z' in suffix:
        plt.ylabel(r'$u$ [m/s]')
        plt.ylim((0, 4.5))
    elif 'v' in suffix:
        plt.ylabel(r'$v$ [m/s]')
        plt.ylim((0, 1.7))
    else:
        plt.ylabel('TKE [$m^2/s^2$]')
        plt.ylim((0, 0.2))

    plt.savefig(vis_path + '/var_over_t' + suffix + '.png', bbox_inches='tight')

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def make_bifurcation_plot_with_Ekman_height(det_data_directory_path, vis_directory_path, param_range):
    # Define colors for plot
    NUM_COLORS = len(param_range) + 1
    cmap = matplotlib.cm.get_cmap('cmc.batlow', NUM_COLORS)
    color = cmap.colors
    norm = matplotlib.colors.BoundaryNorm(param_range, cmap.N)

    # Define figure for plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for idx, var in enumerate(param_range):

        try:
            var = np.around(var, 1)

            # Find simulation file which belongs to current variable (u_G)
            curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]

            # Find Ekman height
            ekman_height = extract_steady_state.find_Ekman_layer_height(det_data_directory_path, vis_directory_path,
                                                                        curr_file_det_sim[0], var, make_plot=False)

            # Remove first 10 hours
            ten_hours_num_points = 60 * 10
            ekman_height = ekman_height[ten_hours_num_points:]

            # Plot Ekman height over geostrophic wind
            ax.scatter(np.repeat(var, len(ekman_height)), ekman_height, label=r'$u_G = $' + str(np.around(var, 1)), s=20, color=color[idx])

        except Exception:
            print(traceback.format_exc())
            pass

    ax.set_xlabel('$u_G$ [m/s]')
    ax.set_ylabel(r'$z_E$ [m]')

    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label=r'$u_G$ [m/s]')

    plt.savefig(vis_directory_path + '/ekman_height_over_uG_all_sim.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    # Define path to deterministic data
    det_directory_path = 'single_column_model/solution/deterministic_94h/'
    det_data_directory_path = det_directory_path + 'simulations/'

    # Create directory to store visualization
    vis_directory_path = os.path.join(det_directory_path, 'visualization')
    if not os.path.exists(vis_directory_path):
        os.makedirs(vis_directory_path)

    # Get a list of all file names in given directory for u and theta
    _, _, files_det = prepare_data.find_files_in_directory(det_data_directory_path)

    param_range = np.arange(1.0, 10.2, 0.2)

    make_bifurcation_plot_with_Ekman_height(det_data_directory_path, vis_directory_path, param_range)

    exit()

    bl_top_height_det_sim_dict, z = prepare_data.find_z_where_u_const(det_data_directory_path, files_det)
    #
    # for var in np.arange(1.0, 6.5, 0.5):
    #     try:
    #         curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]
    #
    #         # Make dataframe of simulation
    #         bl_top_height_det_sim = z[bl_top_height_det_sim_dict[str(var)], :]
    #
    #         df_u, df_v, df_delta_theta, df_tke, _ = create_df_for_fixed_z(det_data_directory_path, curr_file_det_sim,
    #                                                                       bl_top_height_det_sim)
    #
    #         steady_state = find_steady_state(df_u, df_v, df_delta_theta)
    #
    #         # Plot variables over time at BL height
    #         plot_data_over_t(vis_directory_path, df_delta_theta, '_delta_theta_z_const_u_' + str(var), steady_state)
    #         plot_data_over_t(vis_directory_path, df_u, '_u_z_const_u_' + str(var), steady_state)
    #         plot_data_over_t(vis_directory_path, df_v, '_v_z_const_u_' + str(var), steady_state)
    #         #plot_data_over_t(vis_directory_path, df_tke, '_tke_z_const_u_' + str(var))
    #
    #         # # Make 3D plot of increments
    #         # make_3D_plot_increments(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'theta')
    #         # make_3D_plot_increments(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'u')
    #         # make_3D_plot_increments(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'v')
    #
    #         # # Make 3D plot of time series
    #         # make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'theta')
    #         # make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'u')
    #         # make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'v')
    #         # make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'TKE')
    #
    #     except Exception:
    #         print(traceback.format_exc())
    #         break
    #         pass

    # -----------------------------
    # Make bistability plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # color = matplotlib.cm.get_cmap('cmc.batlow', len(np.arange(0.0, 15.5, 0.5) - 1)).colors

    mean_u = []
    mean_delta_theta = []

    param_range = np.arange(2.0, 3.5, 0.1)

    height_idx = 37  # 37, z=20m

    NUM_COLORS = len(param_range) + 1
    cmap = matplotlib.cm.get_cmap('cmc.batlow', NUM_COLORS)
    color = cmap.colors

    norm = matplotlib.colors.BoundaryNorm(param_range, cmap.N)

    for idx, var in enumerate(param_range):

        try:
            var = np.around(var, 1)

            # Define height at which theta_top is calculated
            top_height = 20

            curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]

            # Make dataframe of deterministic simulation
            df_u_det_sim, df_v_det_sim, df_delta_theta_det_sim, _, _ = prepare_data.create_df_for_fixed_z(
                det_data_directory_path,
                curr_file_det_sim,
                top_height)

            steady_state = extract_steady_state.find_steady_state_fixed_height(df_u_det_sim, df_v_det_sim,
                                                                               df_delta_theta_det_sim)
            # print(var)
            # print(steady_state)
            df_u_det_sim = df_u_det_sim.loc[steady_state:steady_state + 60]
            df_delta_theta_det_sim = df_delta_theta_det_sim.loc[steady_state:steady_state + 60]

            mean_u.append(np.mean(df_u_det_sim['sim1']))
            mean_delta_theta.append(np.mean(df_delta_theta_det_sim['sim1']))

            if not np.isnan(mean_u[idx]) and not np.isnan(mean_delta_theta[idx]):
                ax.scatter(df_u_det_sim['sim1'], df_delta_theta_det_sim['sim1'],
                           label=r'$u_G = $' + str(np.around(var, 1)), s=20,
                           color=color[idx])
            # if var == 7.0:
            #     ax.scatter(df_u_det_sim['sim1'], df_delta_theta_det_sim['sim1'], label=r'$u_G = $' + str(var),
            #                color='red', s=20)

        except Exception:
            mean_u.append(np.nan)
            mean_delta_theta.append(np.nan)
            print(traceback.format_exc())
            pass

    # Add line for mean
    # ax.scatter(mean_u, mean_delta_theta, color='grey', s=20)
    ax.plot(mean_u, mean_delta_theta, label='mean', color='grey', linewidth=2, alpha=0.5, linestyle='--')

    # Add vertical lines to indicate transition region
    trans_range_uG = []
    trans_range_mean_u = []

    for idx, val in enumerate(mean_u):
        for small_idx in range(idx)[1:]:
            if mean_u[idx] <= mean_u[small_idx]:
                trans_range_uG.append(param_range[idx])
                trans_range_mean_u.append(mean_u[idx])
        right_interval = [x for x in range(len(mean_u)) if x not in range(idx)][1:]
        for larger_idx in right_interval:
            if mean_u[idx] >= mean_u[larger_idx]:
                trans_range_uG.append(param_range[idx])
                trans_range_mean_u.append(mean_u[idx])

    # Sort lists
    trans_range_mean_u.sort()
    trans_range_uG.sort()
    try:
        plt.axvline(x=trans_range_mean_u[0], color='r', linestyle='--')
        plt.axvline(x=trans_range_mean_u[-1], color='r', linestyle='--')
    except Exception:
        pass

    # ax.set_xlim((2, 7))
    # ax.set_ylim((0, 12))
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel(r'$\Delta \theta$ [K]')
    # plt.legend(ncol=2)

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical',
                        label=r'$u_G$ [m/s]')

    try:
        cbar.ax.axhline(y=trans_range_uG[0], c='r')
        cbar.ax.axhline(y=trans_range_uG[-1], c='r')
    except Exception:
        pass

    plt.savefig(
        vis_directory_path + '/delta_theta_over_u_all_sim_h' + str(int(np.around(z[height_idx, :], 0))) + '.png',
        bbox_inches='tight', dpi=300)
