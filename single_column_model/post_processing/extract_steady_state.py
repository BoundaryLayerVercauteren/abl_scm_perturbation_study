import numpy as np
from functools import reduce
import os
import traceback
import h5py

from single_column_model.post_processing import prepare_data, visualize_deteterministic_model_output


def find_steady_state(data_u, data_v, data_delta_theta):
    """ A steady state is defined as a time point where the value minus a 3 hours average is smaller than 0.1. """

    # Calculate rolling mean
    one_h_num_steps = data_u.index[np.round(data_u['time'], 3) == 1.0].tolist()[0]
    data_u['rol_mean'] = data_u['sim1'].rolling(3 * one_h_num_steps, min_periods=1).mean()
    data_v['rol_mean'] = data_v['sim1'].rolling(3 * one_h_num_steps, min_periods=1).mean()
    data_delta_theta['rol_mean'] = data_delta_theta['sim1'].rolling(3 * one_h_num_steps, min_periods=1).mean()

    # Calculate difference to mean
    data_u['diff_mean'] = np.abs(data_u['rol_mean'] - data_u['sim1'])
    data_v['diff_mean'] = np.abs(data_v['rol_mean'] - data_v['sim1'])
    data_delta_theta['diff_mean'] = np.abs(data_delta_theta['rol_mean'] - data_delta_theta['sim1'])

    # Find all values where the difference to the average is below 0.1
    data_u['bel_thresh'] = data_u['diff_mean'] <= 0.02
    data_v['bel_thresh'] = data_v['diff_mean'] <= 0.02
    data_delta_theta['bel_thresh'] = data_delta_theta['diff_mean'] <= 0.02

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


def extract_initial_cond(curr_steady_state, data_file_path, init_file_path, variable_name):
    with h5py.File(data_file_path, 'r+') as file:
        variable_val = file[variable_name][:]

    initial_cond = variable_val[:, int(curr_steady_state)]

    np.save(init_file_path + variable_name, initial_cond)


if __name__ == '__main__':

    # Define path to deterministic data
    det_directory_path = 'single_column_model/solution/deterministic_84h/'
    det_data_directory_path = det_directory_path + 'simulations/'

    # Create directory to store visualization
    vis_directory_path = os.path.join(det_directory_path, 'visualization')
    if not os.path.exists(vis_directory_path):
        os.makedirs(vis_directory_path)

    # Get a list of all file names in given directory for u and theta
    _, _, files_det = prepare_data.find_files_in_directory(det_data_directory_path)

    bl_top_height_det_sim_dict, z = prepare_data.find_z_where_u_const(det_data_directory_path, files_det)

    for var in np.arange(5.5, 6.0, 0.5):
        try:
            curr_file_det_sim = [s for s in files_det if '_' + str(var) + '_' in s]

            # Make dataframe of simulation
            bl_top_height_det_sim = z[bl_top_height_det_sim_dict[str(var)], :]

            df_u, df_v, df_delta_theta, df_tke, _ = prepare_data.create_df_for_fixed_z(det_data_directory_path,
                                                                                       curr_file_det_sim,
                                                                                       bl_top_height_det_sim)

            steady_state = find_steady_state(df_u, df_v, df_delta_theta)
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


        except Exception:
            print(traceback.format_exc())
            pass
