import numpy as np
import pandas as pd
from functools import reduce
import os
import traceback
import h5py
import matplotlib.pyplot as plt
import cmcrameri.cm as cram

from single_column_model.post_processing import prepare_data, visualize_deteterministic_model_output


def find_steady_state_fixed_height(data_u, data_v, data_delta_theta):
    """ A steady state is defined as a time point where the value minus a 3 hours average is smaller than 0.1. """

    # Calculate rolling mean
    one_h_num_steps = data_u.index[np.round(data_u['time'], 3) == 1.0].tolist()[0]
    data_u['rol_mean'] = data_u['sim1'].rolling(3 * one_h_num_steps, min_periods=1).mean()
    data_v['rol_mean'] = data_v['sim1'].rolling(3 * one_h_num_steps, min_periods=1).mean()
    data_delta_theta['rol_mean'] = data_delta_theta['sim1'].rolling(3 * one_h_num_steps, min_periods=1).mean()

    # Drop first 20 hours
    twenty_h_index = 8 * one_h_num_steps
    data_u = data_u.iloc[twenty_h_index:, :].copy()
    data_v = data_v.iloc[twenty_h_index:, :].copy()
    data_delta_theta = data_delta_theta.iloc[twenty_h_index:, :].copy()

    # Calculate difference to mean
    data_u['diff_mean'] = np.abs(data_u['rol_mean'] - data_u['sim1'])
    data_v['diff_mean'] = np.abs(data_v['rol_mean'] - data_v['sim1'])
    data_delta_theta['diff_mean'] = np.abs(data_delta_theta['rol_mean'] - data_delta_theta['sim1'])

    # Find all values where the difference to the average is below 0.1
    data_u['bel_thresh'] = data_u['diff_mean'] <= data_u['sim1'].max() * 0.02
    data_v['bel_thresh'] = data_v['diff_mean'] <= data_v['sim1'].max() * 0.02
    data_delta_theta['bel_thresh'] = data_delta_theta['diff_mean'] <= data_delta_theta['sim1'].max() * 0.02

    # Find continuous time series where the deviance from the mean is below the threshold
    steady_state = np.nan

    steady_state_range_u = []
    steady_state_range_v = []
    steady_state_range_delta_theta = []

    for k, v in data_u.groupby((data_u['bel_thresh'].shift() != data_u['bel_thresh']).cumsum()):
        if v['bel_thresh'].all() == True:
            if len(v) >= 1.0 * one_h_num_steps:
                steady_state_range_u = np.append(steady_state_range_u, v.index.tolist())

    for k, v in data_v.groupby((data_v['bel_thresh'].shift() != data_v['bel_thresh']).cumsum()):
        if v['bel_thresh'].all() == True:
            if len(v) >= 1.0 * one_h_num_steps:
                steady_state_range_v = np.append(steady_state_range_v, v.index.tolist())

    for k, v in data_delta_theta.groupby(
            (data_delta_theta['bel_thresh'].shift() != data_delta_theta['bel_thresh']).cumsum()):
        if v['bel_thresh'].all() == True:
            if len(v) >= 1.0 * one_h_num_steps:
                steady_state_range_delta_theta = np.append(steady_state_range_delta_theta, v.index.tolist())

    try:
        steady_state = \
            reduce(np.intersect1d, (steady_state_range_u, steady_state_range_v, steady_state_range_delta_theta))[0]
    except Exception:
        pass

    return steady_state


def plot_time_differences(data_path, vis_path, file_name, curr_param, variable_name):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, 'r+') as file:
        # perform byteswap to make handling with pandas dataframe possible
        variable_val = file[variable_name][:].byteswap().newbyteorder()
        z = file['z'][:].byteswap().newbyteorder()
        t = file['t'][:].byteswap().newbyteorder()

    data = pd.DataFrame(data=variable_val.T, columns=z.flatten())
    data['time'] = t.flatten()

    # Calculate rolling mean
    one_h_num_steps_idx = data.index[np.round(data['time'], 3) == 1.0].tolist()[0]
    data_rol_mean = data.rolling(3 * one_h_num_steps_idx, min_periods=1).mean()

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Make plot
    plt.figure(figsize=(5, 5))

    data = data.drop(columns=['time'])
    data_rol_mean = data_rol_mean.drop(columns=['time'])
    plt.pcolor(X, Y, np.abs(data - data_rol_mean).T, cmap=cram.bilbao)  # , vmin=0.0, vmax=1.0)
    plt.axhline(y=z[29, :], color='black', linestyle='--')

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
    elif variable_name == 'TKE':
        cbar.set_label(r'$\tilde{TKE}$', rotation=0)

    # Save plot
    plt.savefig(vis_path + '/3D_plots_increments_' + variable_name + '_' + str(curr_param) + '.png',
                bbox_inches='tight', dpi=300)

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def plot_inversion_strength(data_path, vis_path, file_name, curr_param, steady_state_coord):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, 'r+') as file:
        # perform byteswap to make handling with pandas dataframe possible
        theta = file['theta'][:].byteswap().newbyteorder()
        z = file['z'][:].byteswap().newbyteorder()
        t = file['t'][:].byteswap().newbyteorder()

    theta_df = pd.DataFrame(data=theta.T, columns=z.flatten())

    # Calculate inversion strength
    delta_theta_df = theta_df - theta_df.iloc[0]

    # Calculate height of inversion
    inversion_height_idx = np.zeros((1, len(t.flatten())))
    #inversion_height_idx[...] = np.nan
    for row_idx in delta_theta_df.index[1:]:
        inversion_height_idx[0, row_idx] = np.argmax(np.around(delta_theta_df.iloc[row_idx, :], 1) == 0.0)

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Make plot
    plt.figure(figsize=(5, 5))

    plt.pcolor(X, Y, delta_theta_df.T, cmap=cram.bilbao)
    plt.plot(t.flatten(), z[list(map(int, inversion_height_idx.flatten())), :], color='black', linestyle='--',
             label='inversion height')

    #plt.ylim((1,100))
    plt.xlabel('time [h]')
    plt.ylabel('z [m]')
    cbar = plt.colorbar()
    cbar.set_label(r'$\tilde{\theta}$', rotation=0)

    if steady_state_coord:
        plt.scatter(t.flatten()[int(steady_state_coord[0])], steady_state_coord[1], c='black', s=8, label='steady state', marker='x')

    plt.legend()
    # Save plot
    plt.savefig(vis_path + '/inversion_strength_' + str(curr_param) + '_h' + str(int(np.around(steady_state_coord[1]))) + '.png', bbox_inches='tight', dpi=300)

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def find_Ekman_layer_height(data_path, vis_path, file_name, u_G, steady_state_coord=None):
    full_file_path = data_path + file_name
    # Open output file and load variables
    with h5py.File(full_file_path, 'r+') as file:
        # perform byteswap to make handling with pandas dataframe possible
        u = file['u'][:].byteswap().newbyteorder()
        z = file['z'][:]  # .byteswap().newbyteorder()
        t = file['t'][:]  # .byteswap().newbyteorder()

    data = pd.DataFrame(data=u.T, columns=z.flatten())
    # ata['time'] = t.flatten()

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for row_idx in data.index:
        ekman_height_idx[0, row_idx] = np.argmax(np.around(data.iloc[row_idx, :], 1) == u_G)

    # Create mesh
    X, Y = np.meshgrid(t, z)

    # Make plot
    plt.figure(figsize=(5, 5))
    plt.pcolor(X, Y, u, cmap=cram.davos)
    plt.plot(t.flatten(), z[list(map(int, ekman_height_idx.flatten())), :], color='red', linestyle='--', label='Ekman layer')

    cbar = plt.colorbar()
    cbar.set_label('u', rotation=0, labelpad=2)

    if steady_state_coord:
        plt.scatter(t.flatten()[int(steady_state_coord[0])], steady_state_coord[1], c='black', s=8, label='steady state', marker='x')

    plt.title(r'$u_G = $' + str(u_G))
    plt.xlabel('time [h]')
    plt.ylabel('z [m]')
    plt.legend()

    # Save plot
    plt.savefig(vis_path + '/Ekman_layer_' + str(u_G) + '_h' + str(int(np.around(steady_state_coord[1]))) + '.png', bbox_inches='tight', dpi=300)

    # Clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def extract_initial_cond(curr_steady_state, data_file_path, init_file_path, variable_name):
    with h5py.File(data_file_path, 'r+') as file:
        variable_val = file[variable_name][:]

    initial_cond = variable_val[:, int(curr_steady_state)]

    np.save(init_file_path + variable_name, initial_cond)


if __name__ == '__main__':

    # Define path to deterministic data
    det_directory_path = 'single_column_model/solution/deterministic_94h_zoom_transition/'
    det_data_directory_path = det_directory_path + 'simulations/'

    # Create directory to store visualization
    vis_directory_path = os.path.join(det_directory_path, 'visualization')
    if not os.path.exists(vis_directory_path):
        os.makedirs(vis_directory_path)

    # Get a list of all file names in given directory for u and theta
    _, _, files_det = prepare_data.find_files_in_directory(det_data_directory_path)

    bl_top_height_det_sim_dict, z = prepare_data.find_z_where_u_const(det_data_directory_path, files_det)

    for var in np.arange(2.0, 3.5, 0.1):
        try:
            var = np.around(var, 1)

            curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]

            # # Plot time differences
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'theta')
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'u')
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'v')
            # plot_time_differences(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, 'TKE')

            # Make dataframe of simulation
            height_idx = 37 #37
            bl_top_height_det_sim = z[height_idx,:] #bl_top_height_det_sim_dict[str(var)], :]

            df_u, df_v, df_delta_theta, df_tke, _ = prepare_data.create_df_for_fixed_z(det_data_directory_path,
                                                                                       curr_file_det_sim,
                                                                                       bl_top_height_det_sim)

            steady_state = find_steady_state_fixed_height(df_u, df_v, df_delta_theta)
            print(var)
            print(steady_state)
            # Plot variables over time at BL height
            visualize_deteterministic_model_output.plot_data_over_t(vis_directory_path, df_delta_theta,
                                                                    '_delta_theta_z_const_u_steady_' + str(var),
                                                                    steady_state)
            visualize_deteterministic_model_output.plot_data_over_t(vis_directory_path, df_u,
                                                                    '_u_z_const_u_steady_' + str(var), steady_state)
            visualize_deteterministic_model_output.plot_data_over_t(vis_directory_path, df_v,
                                                                    '_v_z_const_u_steady_' + str(var), steady_state)
            visualize_deteterministic_model_output.plot_data_over_t(vis_directory_path, df_tke,
                                                                    '_tke_z_const_u_steady_' + str(var), steady_state)

            # Extract initial condition
            init_dir_path = 'single_column_model/init_condition/'
            init_file_path = init_dir_path + 'steady_state_Ug' + str(var) + '_'
            extract_initial_cond(steady_state, det_data_directory_path + curr_file_det_sim[0], init_file_path, 'theta')
            extract_initial_cond(steady_state, det_data_directory_path + curr_file_det_sim[0], init_file_path, 'u')
            extract_initial_cond(steady_state, det_data_directory_path + curr_file_det_sim[0], init_file_path, 'v')
            extract_initial_cond(steady_state, det_data_directory_path + curr_file_det_sim[0], init_file_path, 'TKE')

            # Plot inversion strength
            plot_inversion_strength(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, [steady_state, bl_top_height_det_sim[0]])

            # Plot Ekman layer height
            find_Ekman_layer_height(det_data_directory_path, vis_directory_path, curr_file_det_sim[0], var, [steady_state, bl_top_height_det_sim[0]])


        except Exception:
            print(traceback.format_exc())
            pass
