import os
import traceback

import cmcrameri.cm as cram
import h5py
import matplotlib
import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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


def find_files_in_directory(data_path):
    u_files = []
    theta_files = []
    files = []

    for file in os.listdir(data_path):

        if '_u_' in file:
            u_files.append(file)
        elif '_theta_' in file:
            theta_files.append(file)
        elif '.h5' in file:
            files.append(file)

    # Sort files
    u_files = sorted(u_files, key=lambda x: float(x.split('_')[1].strip()))
    theta_files = sorted(theta_files, key=lambda x: float(x.split('_')[1].strip()))
    files = sorted(files, key=lambda x: float(x.split('_')[1].strip()))

    return u_files, theta_files, files


def find_min_max_in_files(directory_path, file_list, variable):
    min_list = []
    max_list = []

    for file in file_list:
        file_path = directory_path + file
        try:
            with h5py.File(file_path, 'r+') as cur_file:
                variable_val = cur_file[variable][:]

                min_list.append(np.min(variable_val))
                max_list.append(np.max(variable_val))

        except Exception:
            pass

    return np.min(min_list), np.max(max_list)


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
    #plt.ylim((0,100))
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
    plt.axhline(y=z[29,:], color='black', linestyle='--')

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


def find_z_where_u_const(data_path, file_paths):
    z_idx_dict = {}
    # Open output file and load variables
    for file_idx, file_path in enumerate(file_paths):
        full_file_path = data_path + file_path

        with h5py.File(full_file_path, 'r+') as file:
            z = file['z'][:]
            u = file['u'][:]

            # Find max and min for every row
            row_max = np.nanmax(u, axis=1)
            row_min = np.nanmin(u, axis=1)
            # Find value range for every row
            row_range = row_max - row_min

            # Find index where z is bigger than 10m and u is near constant
            ten_m_idx = (np.abs(z - 10)).argmin() + 1
            const_u_idx = np.nanargmax(row_range[ten_m_idx:] < 0.3)
            z_idx = const_u_idx + ten_m_idx

            # Set key name
            index_1 = file_path.find('_') + 1
            index_2 = file_path.find('_sim1')
            key_name = file_path[index_1:index_2]
            if z[z_idx, :] > 100:
                z_idx_dict[key_name] = np.nan
            else:
                z_idx_dict[key_name] = z_idx

    # Replace NaN values with mean
    keyList = sorted(z_idx_dict.keys())
    for idx, key in enumerate(z_idx_dict.keys()):
        if np.isnan(z_idx_dict[key]):
            z_idx_dict[key] = z_idx_dict[keyList[idx - 1]]

    return z_idx_dict, z


def create_df_for_fixed_z(data_path, file_paths, height_z):
    # Create empty pandas dataframes
    df_u_temp = {}
    df_v_temp = {}
    df_delta_theta_temp = {}
    df_files_temp = {}
    df_tke_temp = {}

    # Open output file and load variables
    for file_idx, file_path in enumerate(file_paths):
        full_file_path = data_path + file_path

        with h5py.File(full_file_path, 'r+') as file:
            z = file['z'][:]
            t = file['t'][:]

            # Find z which is closest to given value
            z_idx = (np.abs(z - height_z)).argmin()

            # Set name of column
            index_sim = file_path.find('sim')
            index_h5 = file_path.find('.h5')
            column_name = file_path[index_sim:index_h5]

            u = file['u'][:]
            df_u_temp[column_name] = u[z_idx, :]
            df_u_temp[column_name] = df_u_temp[column_name]

            v = file['v'][:]
            df_v_temp[column_name] = v[z_idx, :]
            df_v_temp[column_name] = df_v_temp[column_name]

            theta = file['theta'][:]
            df_delta_theta_temp[column_name] = theta[z_idx, :] - theta[0, :]
            df_delta_theta_temp[column_name] = df_delta_theta_temp[column_name]

            tke = file['TKE'][:]
            df_tke_temp[column_name] = tke[z_idx, :]
            df_tke_temp[column_name] = df_tke_temp[column_name]

            df_files_temp[column_name] = file_path

    df_u = pd.DataFrame({k: list(v) for k, v in df_u_temp.items()})
    df_v = pd.DataFrame({k: list(v) for k, v in df_v_temp.items()})
    df_delta_theta = pd.DataFrame({k: list(v) for k, v in df_delta_theta_temp.items()})
    df_tke = pd.DataFrame({k: list(v) for k, v in df_tke_temp.items()})
    df_files = pd.DataFrame([df_files_temp])

    # Sort columns
    df_u = df_u.reindex(sorted(df_u.columns), axis=1)
    df_v = df_v.reindex(sorted(df_v.columns), axis=1)
    df_delta_theta = df_delta_theta.reindex(sorted(df_delta_theta.columns), axis=1)
    df_tke = df_tke.reindex(sorted(df_tke.columns), axis=1)
    df_files = df_files.reindex(sorted(df_files.columns), axis=1)

    # Add time column to dataframe
    df_u['time'] = t.flatten()
    df_v['time'] = t.flatten()
    df_delta_theta['time'] = t.flatten()
    df_tke['time'] = t.flatten()

    return df_u, df_v, df_delta_theta, df_tke, df_files


def plot_data_over_t(vis_path, data, suffix):
    plt.figure(figsize=(15, 5))

    plt.plot(data['time'], data['sim1'], color='black')

    plt.xlabel('time [h]')
    if 'delta_theta' in suffix:
        plt.ylabel(r'$\Delta \theta$ [K]')
        plt.ylim((0,12))
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


if __name__ == '__main__':

    # Define path to deterministic data
    det_directory_path = 'single_column_model/solution/deterministic_steady_state_check/'
    det_data_directory_path = det_directory_path + 'simulations/'

    # Create directory to store visualization
    vis_directory_path = os.path.join(det_directory_path, 'visualization')
    if not os.path.exists(vis_directory_path):
        os.makedirs(vis_directory_path)

    # Get a list of all file names in given directory for u and theta
    _, _, files_det = find_files_in_directory(det_data_directory_path)

    bl_top_height_det_sim_dict, z = find_z_where_u_const(det_data_directory_path, files_det)

    # for var in np.arange(1.0, 6.5, 0.5):
    #     try:
    #         curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]
    #
    #         # Make dataframe of simulation
    #         bl_top_height_det_sim = z[bl_top_height_det_sim_dict[str(var)], :]
    #
    #         df_u, df_v, df_delta_theta, df_tke, _ = create_df_for_fixed_z(det_data_directory_path, curr_file_det_sim, bl_top_height_det_sim)
    #
    #         # Plot variables over time at BL height
    #         plot_data_over_t(vis_directory_path, df_delta_theta, '_delta_theta_z_const_u_' + str(var))
    #         plot_data_over_t(vis_directory_path, df_u, '_u_z_const_u_' + str(var))
    #         plot_data_over_t(vis_directory_path, df_v, '_v_z_const_u_' + str(var))
    #         plot_data_over_t(vis_directory_path, df_tke, '_tke_z_const_u_' + str(var))
    #
    #         # # Make 3D plot of increments
    #         # make_3D_plot_increments(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'theta')
    #         # make_3D_plot_increments(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'u')
    #         # make_3D_plot_increments(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'v')
    #
    #         #Make 3D plot of time series
    #         make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'theta')
    #         make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'u')
    #         make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'v')
    #         make_3D_plot(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'TKE')
    #
    #     except Exception:
    #         print(traceback.format_exc())
    #         pass

    # -----------------------------
    # Make bistability plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # color = matplotlib.cm.get_cmap('cmc.batlow', len(np.arange(0.0, 15.5, 0.5) - 1)).colors

    mean_u = []
    mean_delta_theta = []

    NUM_COLORS = len(np.arange(1.0, 6.0, 0.5)) + 1
    color = matplotlib.cm.get_cmap('cmc.batlow', NUM_COLORS).colors

    for idx, var in enumerate(np.arange(1.0, 6.0, 0.5)):

        try:
            bl_top_height_det_sim = z[bl_top_height_det_sim_dict[str(var)],:]

            curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]

            # Make dataframe of deterministic simulation
            df_u_det_sim, _, df_delta_theta_det_sim, _, _ = create_df_for_fixed_z(det_data_directory_path,
                                                                                          curr_file_det_sim,
                                                                                          bl_top_height_det_sim)

            df_u_det_sim = df_u_det_sim.loc[1800:]
            df_delta_theta_det_sim = df_delta_theta_det_sim.loc[1800:]

            mean_u.append(np.mean(df_u_det_sim['sim1']))
            mean_delta_theta.append(np.mean(df_delta_theta_det_sim['sim1']))

            if not np.isnan(mean_u[idx]) and not np.isnan(mean_delta_theta[idx]):
                ax.scatter(df_u_det_sim['sim1'], df_delta_theta_det_sim['sim1'], label=r'$u_G = $' + str(var), s=20,
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
    ax.scatter(mean_u, mean_delta_theta, color='grey', s=20)
    ax.plot(mean_u, mean_delta_theta, label='mean', color='grey', linewidth=2)

    # ax.set_xlim((2, 7))
    # ax.set_ylim((0, 12))
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel(r'$\Delta \theta$ [K]')
    plt.legend(ncol=2)
    plt.savefig(vis_directory_path + '/delta_theta_over_u_all_sim_z_const_u.png', bbox_inches='tight', dpi=300)
