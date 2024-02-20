import json
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import scienceplots

# Set plotting style
plt.style.use("science")

# Set font sizes for plots
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define directory where simulation output is saved
output_directory = "results/long_tail/stab_func/gauss_process_stab_func/positive/"
trans_percentage = 0.8

# Load results from sensitivity analysis
with open(f"{output_directory}transition_overview.json") as file:
    result = json.load(file)


# Sort order by wind velocity value and then sigma(s)
def sort_list_of_list(input):
    input.sort(key=itemgetter(2))
    input.sort(key=itemgetter(1))
    input.sort(key=itemgetter(0))
    return input


result = sort_list_of_list(result)

uG_range = np.unique([elem[0] * -1 for elem in result])
sigma_s_range = np.unique([elem[1] for elem in result])
sim_idx_range = np.unique([elem[2] for elem in result])


def average_num_transition_over_all_sim(statistics, uG, sigma_s, sim_idx):
    trans_sum = [
        (
            param_comb,
            sum(
                x[3]
                for x in statistics
                if (x[0] == param_comb[0] and x[1] == param_comb[1])
            ),
        )
        for param_comb in zip(uG, sigma_s)
    ]

    for idx, elem in enumerate(trans_sum):
        trans_sum[idx][2] = trans_sum[idx][2] / np.max(sim_idx) * 100

    return trans_sum


def get_minimal_sigma_with_trans_for_every_u(trans_statistics, u_range):
    first_sigma_with_enough_trans = []
    for u in u_range:
        # Find indices of the parameters which correspond to the current u
        cor_idx = np.where([cur_u[0] for cur_u in trans_statistics] == u)[0]
        # Find out how many transitions (on average) took place for the simulations corresponding to the current u
        cor_average_num_trans = np.array([trans_statistics[i][2] for i in cor_idx])
        idx_first_sigma_with_enough_trans = cor_idx[
            np.argmax(cor_average_num_trans >= trans_percentage)
        ]
        if trans_statistics[idx_first_sigma_with_enough_trans][2] < trans_percentage:
            first_sigma_with_enough_trans.append(np.nan)
        else:
            first_sigma_with_enough_trans.append(
                trans_statistics[idx_first_sigma_with_enough_trans][1]
            )

    return first_sigma_with_enough_trans


min_sigma = np.array(get_minimal_sigma_with_trans_for_every_u(result, uG_range))
print(min_sigma)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# ax.axvspan(5.31, 5.89, alpha=0.3, color="red", label="bistable region")

ax.plot(uG_range, min_sigma, color="darkgreen")
ax.scatter(uG_range, min_sigma, color="darkgreen", marker="v")

# ax.set_xlim((4.5, 6.9))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
# ax.set_axisbelow(True)
ax.grid()

fig.legend(
    facecolor="white",
    edgecolor="black",
    frameon=True,
    bbox_to_anchor=(1.0, 0.95),
    loc="upper left",
)

ax.set_xlabel(r"U [$\mathrm{ms^{-1}}$]")
ax.set_ylabel(r"$\sigma_{s,min} [\mathrm{?}]$")

fig.tight_layout()
plt.savefig(
    f"{output_directory}transition_statistics.pdf", bbox_inches="tight", dpi=300
)
