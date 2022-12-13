import os
import traceback

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cmcrameri.cm as cram  # Package for plot colors
import numpy as np
import pandas as pd
import seaborn as sns

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
        t, _, r, u, v, _, delta_theta, tke, _ = prepare_data.load_data_from_file_for_specific_height(file_path,
                                                                                                     height_z)
        # Skip file if perturbation should be positive bur r is negative
        if 'neg' not in file_path and r < 0:
            continue
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
    # Find delta theta values which are above/below threshold
    if threshold_type == 'above':
        index_values_threshold = np.where(delta_theta_values > regime_threshold)[0]
    elif threshold_type == 'below':
        index_values_threshold = np.where(delta_theta_values < regime_threshold)[0]

    # Find when the regime is exited again !! not correct yet
    entered_regime = False
    index_values_in_regime = []
    for idx in range(len(index_values_threshold))[:-1]:
        if index_values_threshold[idx + 1] - index_values_threshold[idx] == 1:
            index_values_in_regime.append(index_values_threshold[idx])
            if not entered_regime:
                entered_regime = True
        elif entered_regime:
            break
        else:
            continue

    # Return how much time was spend in regime
    return t_values[len(index_values_in_regime)]


def make_heat_map(data, save_directory, plot_name, plot_label, categorical=False):
    # Replace nan with 0
    data_cleaned_up = data.fillna(0)

    # Sort and round index (r values) to allow for better readability of plot
    data_cleaned_up = data_cleaned_up.sort_index()
    data_cleaned_up.index = np.round(data_cleaned_up.index, 4)

    # Change column names to make plot prettier
    data_cleaned_up = data_cleaned_up.rename(
        columns={'u_neg_perturbation': r'$\frac{\partial u}{\partial z} (-)$',
                 'theta_pos_perturbation': r'$\frac{\partial \theta}{\partial z} (+)$',
                 'u_pos_perturbation': r'$\frac{\partial u}{\partial z} (+)$',
                 'theta_neg_perturbation': r'$\frac{\partial \theta}{\partial z} (-)$', })

    # Plot heatmap
    fig, ax = plt.subplots(1, figsize=(5, 5))
    if categorical:
        cmap_heatmap = matplotlib.colors.ListedColormap(['red', 'blue'])
    else:
        cmap_heatmap = plt.get_cmap('rocket_r').copy()
    cmap_heatmap.set_under('white')  # Color for values less than vmin
    heatmap = sns.heatmap(data_cleaned_up.transpose(), ax=ax, cbar_kws={'label': plot_label}, cmap=cmap_heatmap,
                          vmin=0.001)

    # Add border to plot
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('1')

    # Change colorbar tick labels
    if categorical:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(['True', 'False'])

    # Reduce number x axes labels
    df_index = data_cleaned_up.index.to_numpy()
    index_first_tick = np.where(np.round(df_index, 3) == 0.01)[0][0]
    heatmap.set_xticks([index_first_tick * 0.0, index_first_tick, index_first_tick * 2, index_first_tick * 3],
                       [0.0, 0.01, 0.02, 0.03])

    heatmap.set_xlabel(r'max(A(t,z))')
    # Rotate y axes labels
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    plt.savefig('/'.join(save_directory.rsplit('/')[:-4]) + plot_name, bbox_inches='tight', dpi=300)


def make_plot_time_in_new_regime(file_paths, height_z):
    # Find perturbation cases
    perturb_cases = list(set([file_str.rsplit('/')[-3] for file_str in file_paths]))

    # Create dataframe
    transition_duration_df = pd.DataFrame(columns=perturb_cases)

    for idx, file_path in enumerate(file_paths):
        # Load data
        t, _, _, _, _, _, delta_theta, _, max_perturbation = prepare_data.load_data_from_file_for_specific_height(
            file_path, height_z)
        max_perturbation = float(max_perturbation)
        # Calculate time spend in new regime
        if any(np.isnan(delta_theta)):
            continue
        elif 'very_to_weakly' in file_path:
            curr_transition_duration = calculate_time_spend_in_regime(t, delta_theta, 3.0, 'below')
        elif 'weakly_to_very' in file_path:
            curr_transition_duration = calculate_time_spend_in_regime(t, delta_theta, 8.0, 'above')

        # Fill dataframe
        try:
            for case_idx in range(len(perturb_cases)):
                if perturb_cases[case_idx] in file_path:
                    transition_duration_df.at[
                        np.abs(max_perturbation), perturb_cases[case_idx]] = curr_transition_duration
        except Exception:
            print(traceback.format_exc())
            break

    make_heat_map(transition_duration_df, file_paths[0], '/time_in_new_regime_heatmap.png', 'time in regime [h]')


def calculate_time_entered_regime(t_values, delta_theta_values, regime_threshold, threshold_type):
    # Find when regime was entered
    if threshold_type == 'above':
        index_values_threshold = np.where(delta_theta_values > regime_threshold)[0]
    elif threshold_type == 'below':
        index_values_threshold = np.where(delta_theta_values < regime_threshold)[0]

    if len(index_values_threshold) > 0:
        time_in_regime = t_values[index_values_threshold[0]]
    else:
        time_in_regime = 0.0

    # Return time point when regime was entered
    return time_in_regime


def make_plot_transition_time(file_paths, height_z):
    # Find perturbation cases
    perturb_cases = list(set([file_str.rsplit('/')[-3] for file_str in file_paths]))

    # Create dataframe
    transition_duration_df = pd.DataFrame(columns=perturb_cases)

    for idx, file_path in enumerate(file_paths):
        # Load data
        t, _, _, _, _, _, delta_theta, _, max_perturbation = prepare_data.load_data_from_file_for_specific_height(
            file_path, height_z)
        max_perturbation = float(max_perturbation)
        # Calculate time spend in new regime
        if any(np.isnan(delta_theta)):
            continue
        elif 'very_to_weakly' in file_path:
            curr_transition_duration = calculate_time_entered_regime(t, delta_theta, 3.0, 'below')
        elif 'weakly_to_very' in file_path:
            curr_transition_duration = calculate_time_entered_regime(t, delta_theta, 8.0, 'above')

        # Fill dataframe
        try:
            for case_idx in range(len(perturb_cases)):
                if perturb_cases[case_idx] in file_path:
                    transition_duration_df.at[
                        np.abs(max_perturbation), perturb_cases[case_idx]] = curr_transition_duration
        except Exception:
            print(traceback.format_exc())
            break

    make_heat_map(transition_duration_df, file_paths[0], '/transition_time_heatmap.png', 'transition time [h]')


def is_transition_permanent(delta_theta_values, regime_threshold, threshold_type):
    # Find delta theta values which are above/below threshold
    if threshold_type == 'above':
        index_values_in_regime = np.where(delta_theta_values > regime_threshold)[0]
        index_values_outside_regime = np.where(delta_theta_values <= regime_threshold)[0]
    elif threshold_type == 'below':
        index_values_in_regime = np.where(delta_theta_values < regime_threshold)[0]
        index_values_outside_regime = np.where(delta_theta_values >= regime_threshold)[0]

    # Find if the regime is exited again
    exited_regime = 1  # means permanently transitioned (i.e. True)

    if len(index_values_in_regime) > 0:
        if len(index_values_outside_regime) > 0:
            for idx_outside_regime in index_values_outside_regime:
                if idx_outside_regime > np.nanmax(index_values_in_regime):
                    exited_regime = 2  # means not permanently transitioned (i.e. False)
                    break
    else:
        exited_regime = 0  # means regime was never entered

        # Return how much time was spend in regime
    return exited_regime


def make_plot_permanently_transitioned(file_paths, height_z):
    # Find perturbation cases
    perturb_cases = list(set([file_str.rsplit('/')[-3] for file_str in file_paths]))

    # Create dataframe
    transition_duration_df = pd.DataFrame(columns=perturb_cases)

    for idx, file_path in enumerate(file_paths):
        # Load data
        _, _, _, _, _, _, delta_theta, _, max_perturbation = prepare_data.load_data_from_file_for_specific_height(
            file_path, height_z)
        max_perturbation = float(max_perturbation)
        # Classify if the transition was permanent or not
        if any(np.isnan(delta_theta)):
            continue
        elif 'very_to_weakly' in file_path:
            transitioned = is_transition_permanent(delta_theta, 3.0, 'below')
        elif 'weakly_to_very' in file_path:
            transitioned = is_transition_permanent(delta_theta, 8.0, 'above')

        # Fill dataframe
        try:
            for case_idx in range(len(perturb_cases)):
                if perturb_cases[case_idx] in file_path:
                    transition_duration_df.at[
                        np.abs(max_perturbation), perturb_cases[case_idx]] = transitioned
        except Exception:
            print(traceback.format_exc())
            break

    make_heat_map(transition_duration_df, file_paths[0], '/permanently_transitioned_heatmap.png',
                  'permanently transitioned', categorical=True)


def make_plot_crashes(file_paths, height_z):
    # Find perturbation cases
    perturb_cases = list(set([file_str.rsplit('/')[-3] for file_str in file_paths]))

    # Create dataframe
    transition_duration_df = pd.DataFrame(columns=perturb_cases)

    for idx, file_path in enumerate(file_paths):
        # Load data
        _, _, _, _, _, _, delta_theta, _, max_perturbation = prepare_data.load_data_from_file_for_specific_height(
            file_path, height_z)
        max_perturbation = float(max_perturbation)
        # Classify if the transition was permanent or not
        if any(np.isnan(delta_theta)):
            crashed = 1
        else:
            crashed = 2

        # Fill dataframe
        try:
            for case_idx in range(len(perturb_cases)):
                if perturb_cases[case_idx] in file_path:
                    transition_duration_df.at[
                        np.abs(max_perturbation), perturb_cases[case_idx]] = crashed
        except Exception:
            print(traceback.format_exc())
            break

    make_heat_map(transition_duration_df, file_paths[0], '/crashes_heatmap.png',
                  'simulation crashed', categorical=True)


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

    # Make heatmaps to study transition sensitivity
    # make_plot_time_in_new_regime(np.array(all_full_paths).flatten(), height_z_in_m)
    # make_plot_transition_time(np.array(all_full_paths).flatten(), height_z_in_m)
    # make_plot_permanently_transitioned(np.array(all_full_paths).flatten(), height_z_in_m)
    make_plot_crashes(np.array(all_full_paths).flatten(), height_z_in_m)
