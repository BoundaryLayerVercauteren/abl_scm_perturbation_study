import os
import h5py
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as mtransforms
import cmcrameri.cm as cram
from itertools import groupby, count
import seaborn as sns
import numpy as np
import traceback

import pandas as pd

from single_column_model.post_processing import prepare_data

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


def plot_line_transition_plots(vis_path, file_paths, height_z, file_spec):
    # Store r range
    r_range = []

    # Define figure for plot
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

    # Define colors for plot
    NUM_COLORS = len(file_paths) + 1
    cmap = matplotlib.cm.get_cmap('cmc.batlow', NUM_COLORS)
    color = cmap.colors

    for idx, file_path in enumerate(file_paths):
        # Load data
        t, _, r, u, v, _, delta_theta, tke = prepare_data.load_data_from_file_for_specific_height(file_path, height_z)
        # Save r
        r_range.append(r)
        # Extract simulation index
        sim_idx = file_path[-6:-3]
        if 'i' in sim_idx:
            sim_idx = sim_idx[1:]
        if 'm' in sim_idx:
            sim_idx = sim_idx[1:]
        sim_idx = int(sim_idx)
        # Plot solutions over time
        ax[0, 0].plot(t, u, color=color[sim_idx])
        ax[0, 1].plot(t, v, color=color[sim_idx])
        ax[1, 0].plot(t, delta_theta, color=color[sim_idx])
        ax[1, 1].plot(t, tke, color=color[sim_idx])

    # Add labels to subplots
    y_axes_labels = [r'$u$ [m/s]', r'$v$ [m/s]', r'$\Delta \theta$ [K]', 'TKE [$m^2/s^2$]']
    plot_labels = ['a)', 'b)', 'c)', 'd)']

    for ax_idx, single_ax in enumerate(np.ravel(ax)):
        # Add subplot identificator label
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
        single_ax.text(0.0, 1.0, plot_labels[ax_idx], transform=single_ax.transAxes + trans,
                       fontsize='medium', verticalalignment='top')
        # Add y axes label
        single_ax.set_ylabel(y_axes_labels[ax_idx])

    # Improve spacing between subplots
    fig.tight_layout()

    # Add x axes label
    fig.text(0.5, -0.02, 'time [h]', ha='center')

    # Make colorbar
    r_range = np.array(sorted(r_range)).flatten()
    # Reverse colors if any r is negative to much color order in loop
    if any(r < 0.0 for r in r_range):
        cmap = matplotlib.cm.get_cmap('cmc.batlow_r', NUM_COLORS)
    norm = matplotlib.colors.BoundaryNorm(r_range, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('r', rotation=0)

    plt.savefig(vis_path + '/transitions_over_time_uG_' + file_spec + '.png', bbox_inches='tight', dpi=300)

    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def calculate_time_spend_in_regime(t_values, delta_theta_values, regime_threshold, threshold_type):
    # print(delta_theta_values)
    # # Remove first hour
    # index_1_hour = np.where(np.round(t_values,3) == 1.0)[0][0]
    # delta_theta_values = delta_theta_values[index_1_hour:]
    # print(delta_theta_values)
    # Find delta theta values which are above threshold
    if threshold_type == 'above':
        index_values_threshold = np.where(delta_theta_values > regime_threshold)[0]
    elif threshold_type == 'below':
        index_values_threshold = np.where(delta_theta_values < regime_threshold)[0]

    # Find when the regime is exited again !! not correct yet
    entered_regime = False
    index_values_in_regime = []
    for idx in range(len(index_values_threshold))[:-1]:
        if index_values_threshold[idx+1] - index_values_threshold[idx] == 1:
            index_values_in_regime.append(index_values_threshold[idx])
            if not entered_regime:
                entered_regime = True
        elif entered_regime:
            break
        else:
            continue

    # Return how much time was spend in regime
    return t_values[len(index_values_in_regime)]


def make_sensitivity_heat_map(file_paths, height_z):
    # Find perturbation cases
    perturb_cases = list(set([file_str.rsplit('/')[-3] for file_str in file_paths]))

    # Create dataframe
    r_range = np.linspace(0, 0.05, int(len(file_paths) / len(perturb_cases)))
    transition_duration_df = pd.DataFrame(columns=perturb_cases)  # , index=r_range)

    for idx, file_path in enumerate(file_paths):
        # Load data
        t, _, r, _, _, _, delta_theta, _ = prepare_data.load_data_from_file_for_specific_height(file_path, height_z)
        r = float(r)
        # Calculate time spend in new regime
        if 'very_to_weakly' in file_path:
            curr_transition_duration = calculate_time_spend_in_regime(t, delta_theta, 3.0, 'below')
        elif 'weakly_to_very' in file_path:
            curr_transition_duration = calculate_time_spend_in_regime(t, delta_theta, 8.0, 'above')

        # Fill dataframe
        try:
            for case_idx in range(len(perturb_cases)):
                if perturb_cases[case_idx] in file_path:
                    transition_duration_df.at[np.abs(r), perturb_cases[case_idx]] = curr_transition_duration
        except Exception:
            print(traceback.format_exc())
            break

    # Replace nan with 0
    transition_duration_df_no_nan = transition_duration_df.fillna(0)

    # Sort and round index (r values) to allow for better readability of plot
    transition_duration_df_no_nan = transition_duration_df_no_nan.sort_index()
    transition_duration_df_no_nan.index = np.round(transition_duration_df_no_nan.index, 4)

    # Plot heatmap
    fig, ax = plt.subplots(1, figsize=(5, 5))
    heatmap = sns.heatmap(transition_duration_df_no_nan.transpose(), ax=ax, cbar_kws={'label': 'time in regime [h]'}, cmap='rocket_r')

    # Reduce number x axes labels
    df_index = transition_duration_df_no_nan.index.to_numpy()
    index_first_tick = np.where(np.round(df_index, 3) == 0.01)[0][0]
    heatmap.set_xticks([index_first_tick, index_first_tick * 2, index_first_tick * 3, index_first_tick * 4],
                       [0.01, 0.02, 0.03, 0.04])

    heatmap.set_xlabel('r')
    plt.savefig('/'.join(file_paths[0].rsplit('/')[:-4]) + '/transitions_heatmap.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    # Define path to perturbed data
    data_directory_path_top = 'single_column_model/solution/perturbed/'
    data_directory_path_sub = ['weakly_to_very_stable/u_neg_perturbation/',
                               'weakly_to_very_stable/theta_pos_perturbation/',
                               'very_to_weakly_stable/theta_neg_perturbation/',
                               'very_to_weakly_stable/u_pos_perturbation/']

    all_full_paths = []

    # Set height at which transition are studied
    height_z_in_m = 20

    for idx_perturb_type in range(len(data_directory_path_sub)):

        data_directory_path = data_directory_path_top + data_directory_path_sub[idx_perturb_type]

        # Define path to simulation output files
        data_directory_path_sim = data_directory_path + 'simulations/'

        # Create directory to store visualizations
        vis_directory_path = os.path.join(data_directory_path, 'visualization')
        if not os.path.exists(vis_directory_path):
            os.makedirs(vis_directory_path)

        # Get a list of all file names in given directory
        _, _, files_sim = prepare_data.find_files_in_directory(data_directory_path_sim)

        # Get u_G range
        uG_range = list(set([float(sub_str[9:12]) for sub_str in files_sim]))

        uG = np.around(uG_range[0], 1)

        # Get all files which correspond to current uG
        curr_files_single_sim = [s for s in files_sim if '_' + str(uG) + '_' in s]
        full_path_current_sim = sorted([data_directory_path_sim + file for file in curr_files_single_sim])
        all_full_paths.append(full_path_current_sim)

        # Plot transitions over time
        # plot_line_transition_plots(vis_directory_path, full_path_current_sim, height_z_in_m, file_spec=str(uG))

    # Make heatmap to study transition sensitivity
    make_sensitivity_heat_map(np.array(all_full_paths).flatten(), height_z_in_m)
