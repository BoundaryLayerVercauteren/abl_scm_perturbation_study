"""Needs to be run with: python3 -m single_column_model.post_processing.deterministic_data.plot_stability_function"""

import matplotlib.pyplot as plt
import numpy as np

from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(5, 5))


def define_delage_short_tail_stab_function(Ri):
    return 1 + 12 * Ri


def define_delage_long_tail_stab_function(Ri):
    return 1 + 4.7 * Ri


def define_cut_off_costa_stab_function(Ri):
    if Ri < 0.25:
        return (1 - Ri / 0.25) ** 2
    else:
        return 0


richardson_num = np.linspace(10 ** (-4), 10 ** (1), 1000)

vec_delage_short_tail_stab_func = np.vectorize(define_delage_short_tail_stab_function)
vec_delage_long_tail_stab_func = np.vectorize(define_delage_long_tail_stab_function)
vec_cut_off = np.vectorize(define_cut_off_costa_stab_function)


fig, ax = plt.subplots(1, figsize=[5, 5])

ax.plot(
    richardson_num,
    1 / vec_delage_short_tail_stab_func(richardson_num),
    label=r"1/$1 + 12$  Ri",
    color="black",
)

ax.plot(
    richardson_num,
    1 / vec_delage_long_tail_stab_func(richardson_num),
    label=r"1/$1 + 4.7$  Ri",
    color="red",
)

# ax.plot(
#     richardson_num,
#     1/vec_cut_off(richardson_num),
#     label=r"cut_off",
#     color="blue",
# )

# ax.set_xscale("log")
ax.set_xlabel("Ri")
ax.set_ylabel(r"1/$\phi$")

# ax.set_ylim(0,1)

plt.legend()

plt.savefig(
    "single_column_model/solution/stability_function.png", bbox_inches="tight", dpi=300
)
