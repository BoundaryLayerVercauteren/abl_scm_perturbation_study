"""Needs to be run with: python3 -m single_column_model.post_processing.deterministic_data.extract_steady_state"""

import warnings
from functools import reduce

warnings.simplefilter(action="ignore", category=FutureWarning)

import h5py
import numpy as np
import matplotlib.pyplot as plt

from single_column_model.post_processing.deterministic_data import prepare_data
from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(10, 5))


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
        data_delta_theta["sim_0"]
        .rolling(num_hours * one_h_num_steps, min_periods=1)
        .mean()
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
    deviation_percentage = 0.05
    data_u["bel_thresh"] = data_u["diff_mean"] <= deviation_percentage
    data_v["bel_thresh"] = data_v["diff_mean"] <= deviation_percentage
    data_delta_theta["bel_thresh"] = (
        data_delta_theta["diff_mean"] <= deviation_percentage
    )
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


def save_initial_cond_for_single_variable_to_file(
    curr_steady_state, data_file_path, init_file_path, variable_name
):
    with h5py.File(data_file_path, "r+") as file:
        variable_val = file[variable_name][:]

    initial_cond = variable_val[:, int(curr_steady_state)]
    np.save(init_file_path + variable_name, initial_cond)


def save_initial_cond_for_all_variable_to_file(
    stab_func_type, uG_value, steady_state, data_path
):
    init_dir_path = "single_column_model/init_condition/"
    init_file_path = init_dir_path + f"{stab_func_type}_steady_state_Ug{uG_value}_"

    variables = ["theta", "u", "v", "TKE"]

    for var in variables:
        save_initial_cond_for_single_variable_to_file(
            steady_state, data_path, init_file_path, var
        )


def plot_steady_state_for_all_variables(
    vis_path, uG, data_u, data_v, data_delta_theta, data_tke, steady_state
):

    steady_state = int(steady_state)

    fig, ax = plt.subplots(2, 2, figsize=[10, 5], sharex=True)
    ax = ax.ravel()

    time_range = [0, steady_state + 10 * 60]

    ax[0].plot(
        data_u.iloc[time_range[0] : time_range[1], 1],
        data_u.iloc[time_range[0] : time_range[1], 0],
        color="black",
    )
    ax[1].plot(
        data_delta_theta.iloc[time_range[0] : time_range[1], 1],
        data_delta_theta.iloc[time_range[0] : time_range[1], 0],
        color="black",
    )
    ax[2].plot(
        data_v.iloc[time_range[0] : time_range[1], 1],
        data_v.iloc[time_range[0] : time_range[1], 0],
        color="black",
    )
    ax[3].plot(
        data_tke.iloc[time_range[0] : time_range[1], 1],
        data_tke.iloc[time_range[0] : time_range[1], 0],
        color="black",
    )

    ax[0].axvspan(
        data_u.iloc[steady_state, 1],
        data_u.iloc[60 + steady_state, 1],
        color="red",
        alpha=0.5,
        label="steady state region",
    )
    ax[1].axvspan(
        data_delta_theta.iloc[steady_state, 1],
        data_delta_theta.iloc[60 + steady_state, 1],
        color="red",
        alpha=0.5,
        label="steady state region",
    )
    ax[2].axvspan(
        data_v.iloc[steady_state, 1],
        data_v.iloc[60 + steady_state, 1],
        color="red",
        alpha=0.5,
        label="steady state region",
    )
    ax[3].axvspan(
        data_tke.iloc[steady_state, 1],
        data_tke.iloc[60 + steady_state, 1],
        color="red",
        alpha=0.5,
        label="steady state region",
    )

    ax[2].set_xlabel("time [h]")
    ax[3].set_xlabel("time [h]")

    ax[0].set_ylabel(r"$u \ [\mathrm{ms^{-1}}]$")
    ax[1].set_ylabel(r"$\Delta \theta$ [K]")
    ax[2].set_ylabel(r"$v \ [\mathrm{ms^{-1}}]$")
    ax[3].set_ylabel(r"TKE [$\mathrm{m^2s^{-2}}$]")

    fig.tight_layout()

    plt.savefig(f"{vis_path}/steady_state_uG_{uG}.png", bbox_inches="tight", dpi=300)

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close("all")  # Closes all the figure windows.


# Define path to data for which the steady state shall be determined
stab_func_type = "long_tail"
data_path = f"single_column_model/solution/{stab_func_type}/deterministic/simulations/"
vis_path = f"single_column_model/solution/{stab_func_type}/deterministic/visualization/"

uG_range = np.arange(1.0, 2.5, 0.1)
steady_state_height = 20  # in meters

# Get a list of all solution file names in given directory
data_files = prepare_data.find_solution_files_in_directory(data_path)

for uG in uG_range:

    uG = np.round(uG, 1)

    # Find file which corresponds to uG
    curr_file = [s for s in data_files if f"_{uG}_" in s]

    try:
        # Get simulation data for uG
        df_u, df_v, df_delta_theta, df_tke, _ = prepare_data.create_df_for_fixed_z(
            data_path, curr_file, steady_state_height
        )
    except Exception:
        print(f"error for uG={uG}")
        continue

    # Determine steady state
    steady_state = find_steady_state_fixed_height(df_u, df_v, df_delta_theta, df_tke)

    if np.isnan(steady_state):
        print(f"for ugG={uG} no steady state was found.")
        plot_steady_state_for_all_variables(
            vis_path, uG, df_u, df_v, df_delta_theta, df_tke, df_u.shape[0] - 600
        )
        continue

    # Save steady state as initial condition
    save_initial_cond_for_all_variable_to_file(
        stab_func_type, uG, steady_state, data_path + curr_file[0]
    )

    # Plot steady state
    plot_steady_state_for_all_variables(
        vis_path, uG, df_u, df_v, df_delta_theta, df_tke, steady_state
    )
