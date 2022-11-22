import os
import numpy as np
import pandas as pd
import h5py


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


def create_df_for_fixed_z(data_path, file_paths, height_z, file_type='deterministic'):
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
            if file_type == 'deterministic':
                index_sim = file_path.find('sim')
                index_h5 = file_path.find('.h5')
                column_name = file_path[index_sim:index_h5]
            else:
                r = file['r'][:][0][0]
                column_name = str(r)

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
