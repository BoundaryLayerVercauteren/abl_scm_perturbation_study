import h5py
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cram

tau_list = [2, 5, 8]
uG = 2.2
perturbed_param = "u"

colors = ["red", "orange", "blue"]

# fig, ax = plt.subplots(1, 1, figsize=[10, 5])
#
# for idx, tau in enumerate(tau_list):
#     file_path = f'single_column_model/solution/short_tail/nudging_test/tau_{tau}/perturbed_{perturbed_param}/simulations/solution_uG_{uG}_perturbstr_0.015_sim_0.0.h5'
#
#     with h5py.File(file_path, "r+") as file:
#         z = file["z"][:]
#         t = file["t"][:]
#         z_idx = (np.abs(z - 20)).argmin()
#         theta = file["theta"][:]
#         print(np.nanmax(file['perturbation'][:]))
#
#         delta_theta = theta[z_idx, :] - theta[0, :]
#
#     ax.plot(t.flatten(), delta_theta.flatten(), color=colors[idx], label=tau)
#
# ax.set_xlabel("time [h]")
# ax.set_ylabel(r"$\Delta \theta$ [K]")
# ax.legend()
# fig.tight_layout()
#
# plt.savefig(f"single_column_model/solution/short_tail/nudging_test/visualization/perturbed_sol_{uG}_all_tau_{perturbed_param}.png", bbox_inches="tight", dpi=300)

fig, ax = plt.subplots(1, 3, figsize=[15, 5])
ax = ax.ravel()

for idx, tau in enumerate(tau_list):
    file_path = f"single_column_model/solution/short_tail/nudging_test/tau_{tau}/perturbed_{perturbed_param}/simulations/solution_uG_{uG}_perturbstr_0.015_sim_0.0.h5"

    with h5py.File(file_path, "r+") as file:
        z = file["z"][:]
        t = file["t"][:]
        z_idx = (np.abs(z - 20)).argmin()
        u = file["u"][:]
        v = file["v"][:]
        wind_direction = np.arctan2(v, u) * 180 / np.pi

    z_max = (np.abs(z - 100)).argmin()
    t_max = 120
    X, Y = np.meshgrid(t[0, :t_max], z[:z_max, 0])

    im = ax[idx].pcolor(
        X, Y, wind_direction[:z_max, :t_max], cmap=cram.davos, vmin=0, vmax=180
    )
    ax[idx].set_xlabel("time [h]")
    ax[idx].set_ylabel("z [m]")
    ax[idx].title.set_text(r"$\tau = $" + str(tau))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# fig.tight_layout()

plt.savefig(
    f"single_column_model/solution/short_tail/nudging_test/visualization/3D_wind_dir_perturbed_sol_{uG}_all_tau_{perturbed_param}.png",
    bbox_inches="tight",
    dpi=300,
)
