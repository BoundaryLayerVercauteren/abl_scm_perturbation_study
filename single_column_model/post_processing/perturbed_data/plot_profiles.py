import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import h5py
import cmcrameri.cm as cmc

from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(10, 10))

uGs = [1.6, 1.9]  # np.round(np.arange(1.6, 2.2, 0.1),1)
time_h = np.arange(0, 6, 1)

sim_dir = "/mn/vann/amandink/02_sbl_single_column_model/output/short_tail/"


def get_profiles(file_path, time_idx):
    # Open output file and load variables
    with h5py.File(file_path, "r+") as file:
        u = file["u"][:]
        v = file["v"][:]
        z = file["z"][:]
        t = file["t"][:]
        theta = file["theta"][:]

    row_idx = time_idx * 60
    wind_speed = np.sqrt(u**2 + v**2)

    return wind_speed[:, row_idx], theta[:, row_idx], z.flatten()


def find_Ekman_layer_height(file_path, u_G):
    # Open output file and load variables
    with h5py.File(file_path, "r+") as file:
        u = file["u"][:]
        v = file["v"][:]
        z = file["z"][:]
        t = file["t"][:]

    wind_speed = np.sqrt(u**2 + v**2)

    data = pd.DataFrame(data=wind_speed.T, columns=z.flatten())

    lower_z = [col for col in data.columns if col < 5]
    data.loc[:, lower_z] = np.nan

    ekman_height_idx = np.zeros((1, len(t.flatten())))
    ekman_height_idx[...] = np.nan
    for time_idx in data.index:
        ekman_height_idx[0, time_idx] = np.argmax(
            np.isclose(data.iloc[time_idx, :], u_G, atol=2e-1)
        )

    return (
        z[list(map(int, ekman_height_idx.flatten())), :].flatten(),
        t.flatten(),
        z.flatten(),
    )


def make_line_plot(data, axes, title, z, all_axes=[]):

    columns = data.columns.astype(float)

    # axes.scatter(np.repeat(12, len(z)), z, color='black', s=10)

    # for column in data.columns:
    #     if column not in [1.0, 1.7, 1.8, 2.3]:
    #         data = data.drop(column, axis=1)

    sm = plt.cm.ScalarMappable(
        cmap="cmc.roma", norm=plt.Normalize(vmin=columns.min(), vmax=columns.max())
    )
    sm._A = []

    plot = data.plot(kind="line", legend=False, ax=axes, colormap="cmc.roma", zorder=10)

    for line in axes.get_lines():
        if line.get_label() == "1.6" or line.get_label() == "1.9":
            line.set_linewidth(5)

    if title == "d)":
        cbar = plt.colorbar(sm, ax=all_axes, location="right", shrink=0.8)
        cbar.set_label(r"$s_G \ [\mathrm{ms^{-1}}]$")

    NUM_COLORS = 7
    cmap = matplotlib.cm.get_cmap("cmc.hawaii", NUM_COLORS)
    color = cmap.colors
    time_idx = np.arange(0, 6, 1)
    for idx in time_idx:
        axes.axvline(x=idx, color=color[idx], linestyle=":", zorder=0)

    # axes.set_xlim((0,1.5))
    axes.set_ylim((0, 40))
    axes.set_ylabel("Ekman layer height [m]")
    # axes.set_yticklabels([])
    axes.set_xlabel(r"t [h]")
    axes.set_title(title, loc="left")


def get_all_Ekman_layer_heights(theta_perturb_strength, uG_list, perturbation_type):
    ekman_height_theta = {}
    ekman_height_u = {}

    for idx, uG in enumerate(uG_list):
        if not np.isnan(theta_perturb_strength[idx]):
            file_path_theta = f"{sim_dir}/dt_1/300_5/{perturbation_type}/simulations/solution_uG_{uG}_perturbstr_{theta_perturb_strength[idx]}_sim_0.h5"

            ekman_height_theta[uG], time, z = find_Ekman_layer_height(
                file_path_theta, uG
            )
        else:
            ekman_height_theta[uG] = np.repeat(np.nan, 720).flatten()

    df_ekman_layer_height_theta = pd.DataFrame(
        {k: list(v) for k, v in ekman_height_theta.items()}
    )

    df_ekman_layer_height_theta = df_ekman_layer_height_theta.reindex(
        sorted(df_ekman_layer_height_theta.columns), axis=1
    )

    df_ekman_layer_height_theta = df_ekman_layer_height_theta.set_index(time)

    # Drop ten hours
    ten_h_idx = 10
    df_ekman_layer_height_theta[ten_h_idx:]

    return df_ekman_layer_height_theta, z


NUM_COLORS = 7
cmap = matplotlib.cm.get_cmap("cmc.hawaii", NUM_COLORS)
color = cmap.colors

titles = ["a)", "b)", "c)", "d)", "e)", "f)"]

fig, ax = plt.subplots(
    2, 3, figsize=(15, 10), constrained_layout=True
)  # , sharex=True, sharey=True)
ax = ax.ravel()

idx = 0

perturbation_var = "theta"
uG_list = np.round(np.arange(1.0, 2.6, 0.1), 1)
theta_perturb_strength = np.repeat(0.019, 17)
Ekman_height_df, z = get_all_Ekman_layer_heights(
    theta_perturb_strength, uG_list, "neg_theta"
)
make_line_plot(Ekman_height_df, ax[idx], titles[idx], z)
idx += 1

for uG in uGs:
    directory = f"{sim_dir}/dt_1/300_5/neg_theta/simulations/solution_uG_{uG}_perturbstr_0.019_sim_0.h5"
    wind, temp, z = get_profiles(directory, time_h)

    for row in np.arange(0, np.shape(wind)[1]):
        ax[idx].plot(temp[:, row], z, label=time_h[row], color=color[row])

    ax[idx].set_xlabel(r"$\theta$ [K]")
    ax[idx].set_ylabel("z [h]")
    ax[idx].set_ylim((0, 100))
    ax[idx].set_xlim((275, 305))
    ax[idx].legend()
    ax[idx].set_title(titles[idx], loc="left")
    idx += 1

Ekman_height_df, z = get_all_Ekman_layer_heights(
    theta_perturb_strength, uG_list, "pos_theta"
)
make_line_plot(Ekman_height_df, ax[idx], titles[idx], z, [ax[0], ax[idx]])
idx += 1

for uG in uGs:
    directory = f"{sim_dir}/dt_1/300_5/pos_theta/simulations/solution_uG_{uG}_perturbstr_0.019_sim_0.h5"
    wind, temp, z = get_profiles(directory, time_h)

    for row in np.arange(0, np.shape(wind)[1]):
        ax[idx].plot(temp[:, row], z, label=time_h[row], color=color[row])

    ax[idx].set_xlabel(r"$\theta$ [K]")
    ax[idx].set_ylabel("z [h]")
    ax[idx].set_ylim((0, 100))
    ax[idx].set_xlim((285, 315))
    ax[idx].legend()
    ax[idx].set_title(titles[idx], loc="left")
    idx += 1


cols = ["", r"$s_G=1.6 \ [\mathrm{ms^{-1}}]$", r"$s_G=1.9 \ [\mathrm{ms^{-1}}]$"]
rows = ["cold air advection", "", "", "warm air advection"]
pad = 5  # in points
for axes, col in zip(ax, cols):
    axes.annotate(
        col,
        xy=(0.5, 1),
        xytext=(0, pad),
        xycoords="axes fraction",
        textcoords="offset points",
        size="large",
        ha="center",
        va="baseline",
    )

for axes, row in zip(ax, rows):
    if row != "":
        axes.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-axes.yaxis.labelpad - pad, 0),
            xycoords=axes.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation=90,
        )


# fig.tight_layout()
# fig.subplots_adjust(left=0.15, top=0.95)
plt.savefig(
    f"{sim_dir}/dt_1/visualization/temperature_profiles.png",
    bbox_inches="tight",
    dpi=300,
)
