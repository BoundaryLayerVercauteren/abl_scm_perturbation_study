import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(10, 10))


det_perturb_file = "single_column_model/solution/short_tail/perturbed/pos_theta/simulations/solution_uG_1.0_perturbstr_0.019_sim_0.h5"


def get_stoch_stab_func_data(full_file_path, z_max=40, t_max=12):
    with h5py.File(full_file_path, "r+") as file:
        phi = file["phi"][:]
        t = file["t"][:]
        z = file["z"][:]

    z_idx_max = (np.abs(z - z_max)).argmin()
    t_idx_max = (np.abs(t - t_max)).argmin()

    T, Z = np.meshgrid(t[0, :t_idx_max], z[:z_idx_max, :])

    return phi[:z_idx_max, :t_idx_max], T, Z


def get_perturbation_data(full_file_path, z_max=40, t_max=12):
    with h5py.File(full_file_path, "r+") as file:
        perturbation = file["perturbation"][:]
        t = file["t"][:]
        z = file["z"][:]

    z_idx_max = (np.abs(z - z_max)).argmin()
    t_idx_max = (np.abs(t - t_max)).argmin()

    T, Z = np.meshgrid(t[0, :t_idx_max], z[:z_idx_max, :])

    return perturbation[:z_idx_max, :t_idx_max], T, Z


fig = plt.figure(figsize=(10, 5), layout="constrained")
ax = fig.subplot_mosaic(
    [["ssf1", "ssf2", "perturbation"], ["ssf3", "ssf4", "perturbation"]],
    width_ratios=[1, 1, 2],
    sharex=True,
    sharey=True,
)

sub_plot_labels = ["ssf1", "ssf2", "ssf3", "ssf4", "perturbation"]
levels = np.linspace(0.0, 20.0, 6)
for idx, sim_idx in enumerate([14, 40, 60, 80]):
    stoch_stab_func_file = f"single_column_model/solution/short_tail/perturbed/stab_func/simulations/solution_uG_1.0_perturbstr_1.0_sim_{sim_idx}.0.h5"
    stoch_stab_func_data, time_sf_grid, space_sf_grid = get_stoch_stab_func_data(
        stoch_stab_func_file
    )
    stoch_stab_plot = ax[sub_plot_labels[idx]].contourf(
        time_sf_grid,
        space_sf_grid,
        stoch_stab_func_data,
        levels=levels,
        cmap=cmc.lipari_r,
    )

    if idx > 1:
        ax[sub_plot_labels[idx]].set_xlabel("t [h]")
    if idx == 0 or idx == 2:
        ax[sub_plot_labels[idx]].set_ylabel("z [m]")

cbar_stoch_stab = plt.colorbar(
    stoch_stab_plot,
    ax=[ax["ssf1"], ax["ssf2"], ax["ssf3"], ax["ssf4"]],
    location="bottom",
)  # , shrink=0.8)
cbar_stoch_stab.set_label(r"$\rho$")

perturbation_data, time_p_grid, space_p_grid = get_perturbation_data(det_perturb_file)
levels = np.linspace(0.0, 0.02, 6)
perturbation_plot = ax["perturbation"].contourf(
    time_p_grid, space_p_grid, perturbation_data, levels=levels, cmap=cmc.lapaz
)
cbar_perturbation = plt.colorbar(
    perturbation_plot, ax=ax["perturbation"], location="bottom"
)  # , shrink=0.8)
cbar_perturbation.set_label(r"$p$")
ax["perturbation"].set_xlabel("t [h]")
ax["perturbation"].yaxis.set_tick_params(labelbottom=True)
ax["perturbation"].set_ylabel("z [m]")

ax["ssf1"].set_title("a)", loc="left")
ax["perturbation"].set_title("b)", loc="left")

plt.figtext(0.27, 0.98, "stochastic stability function", va="center", ha="center")
plt.figtext(0.77, 0.98, "deterministic perturbation", va="center", ha="center")


plt.savefig(
    "single_column_model/solution/short_tail/perturbed/stoch_stab_func_vs_det_perturbation.png",
    bbox_inches="tight",
    dpi=300,
)
