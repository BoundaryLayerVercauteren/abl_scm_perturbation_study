import os

import h5py
import numpy as np
import pandas as pd


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
    u_files = sorted(u_files, key=lambda x: float(x.split("_")[2].strip()))
    theta_files = sorted(theta_files, key=lambda x: float(x.split("_")[2].strip()))
    files = sorted(files, key=lambda x: float(x.split("_")[2].strip()))

    return u_files, theta_files, files


def load_data_from_file_for_specific_height(file_paths, height_z):
    # Load data
    with h5py.File(file_paths, "r+") as file:
        t = file["t"][:]
        z = file["z"][:]
        r = file["r"][:]
        u = file["u"][:]
        v = file["v"][:]
        theta = file["theta"][:]
        tke = file["TKE"][:]
        perturbation = file["perturbation"][:]

    # Find z which is closest to given value
    z_idx = (np.abs(z - height_z)).argmin()

    # Get data for specific height
    u_height_z = u[z_idx, :].flatten()
    v_height_z = v[z_idx, :].flatten()
    theta_height_z = theta[z_idx, :].flatten()
    delta_theta = theta[z_idx, :] - theta[0, :].flatten()
    tke_height_z = tke[z_idx, :].flatten()

    # maximal perturbation is defined as the absolut maximal perturbation
    if r[0] >= 0:
        max_perturbation = np.nanmax(perturbation)
    else:
        max_perturbation = np.nanmin(perturbation)

    z_idx_100m = (np.abs(z - 100)).argmin()
    max_u = np.nanmax(np.abs(u[:z_idx_100m, 0]))
    max_theta = np.nanmax(np.abs(theta[:z_idx_100m, 0]))

    return (
        t.flatten(),
        z.flatten(),
        r[0][0],
        u_height_z,
        v_height_z,
        theta_height_z,
        delta_theta,
        tke_height_z,
        max_perturbation / max_u,
        max_perturbation / max_theta,
    )


def create_df_for_fixed_z(data_path, file_paths, height_z, file_type="deterministic"):
    # Create empty pandas dataframes
    df_u_temp = {}
    df_v_temp = {}
    df_delta_theta_temp = {}
    df_files_temp = {}
    df_tke_temp = {}

    # Open output file and load variables
    for file_idx, file_path in enumerate(file_paths):
        full_file_path = data_path + file_path

        with h5py.File(full_file_path, "r+") as file:

            z = file["z"][:]
            t = file["t"][:]

            # Find z which is closest to given value
            z_idx = (np.abs(z - height_z)).argmin()

            # Set name of column
            if file_type == "deterministic":
                index_sim = file_path.find("sim")
                index_h5 = file_path.find(".h5")
                column_name = file_path[index_sim:index_h5]
            else:
                r = file["r"][:][0][0]
                column_name = r

            u = file["u"][:]
            df_u_temp[column_name] = u[z_idx, :]
            df_u_temp[column_name] = df_u_temp[column_name]

            v = file["v"][:]
            df_v_temp[column_name] = v[z_idx, :]
            df_v_temp[column_name] = df_v_temp[column_name]

            theta = file["theta"][:]
            df_delta_theta_temp[column_name] = theta[z_idx, :] - theta[0, :]
            df_delta_theta_temp[column_name] = df_delta_theta_temp[column_name]

            tke = file["TKE"][:]
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
    df_u["time"] = t.flatten()
    df_v["time"] = t.flatten()
    df_delta_theta["time"] = t.flatten()
    df_tke["time"] = t.flatten()

    return df_u, df_v, df_delta_theta, df_tke, df_files
