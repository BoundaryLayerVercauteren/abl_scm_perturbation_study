"""Needs to be run with: python3 -m single_column_model.post_processing.deterministic_data.plot_stability_function"""

import matplotlib.pyplot as plt
import numpy as np

from single_column_model.post_processing import set_plotting_style

set_plotting_style.set_style_of_plots(figsize=(5, 5))


def define_delage_short_tail_stab_function(Ri):
    return 1 + 12 * Ri


richardson_num = np.linspace(10 ** (-4), 10 ** (1), 1000)

vec_delage_short_tail_stab_func = np.vectorize(define_delage_short_tail_stab_function)

fig, ax = plt.subplots(1, figsize=[5, 5])

ax.plot(
    richardson_num,
    vec_delage_short_tail_stab_func(richardson_num),
    label=r"$1 + 12$  Ri",
    color="black",
)

ax.set_xscale("log")
ax.set_xlabel("Ri")
ax.set_ylabel(r"$\phi$")

plt.legend()

plt.savefig(
    "single_column_model/solution/stability_function.png", bbox_inches="tight", dpi=300
)
