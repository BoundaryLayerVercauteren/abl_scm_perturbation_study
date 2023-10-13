import cmcrameri as cram  # Package for plot colors
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

#plt.style.use("science")

# set font sizes for plots
SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def define_vandewiel_short_tail_exp_stab_function(Ri, alpha=5):
    return np.exp(-2 * alpha * Ri - (alpha * Ri) ** 2)


def define_vandewiel_short_tail_stab_function(Ri, alpha=5):
    return (1 - alpha * Ri) ** 2


def define_delage_long_tail_stab_function(Ri):
    return 1 + 12 * Ri


def define_delage_short_tail_stab_function(Ri):
    return 1 + 4.7 * Ri


def make_comparison():
    richardson_num = np.linspace(10 ** (-4), 10 ** (1), 1000)

    vec_delage_short_tail_stab_func = np.vectorize(define_delage_short_tail_stab_function)
    vec_delage_long_tail_stab_func = np.vectorize(define_delage_long_tail_stab_function)

    # Create plot
    color = matplotlib.cm.get_cmap("cmc.batlow", 4).colors
    markers = ["v", "*", "^", "s", "p", "."]

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(
        richardson_num,
        vec_delage_long_tail_stab_func(richardson_num),
        label=r"$1 + 12  Ri$ (long)",
        color=color[1],
        marker=markers[1],
        markevery=100,
    )
    ax1.plot(
        richardson_num,
        vec_delage_short_tail_stab_func(richardson_num),
        label=r"$1 + 4.7  Ri$ (short)",
        color=color[2],
        marker=markers[2],
        markevery=100,
    )

    ax1.set_xscale('log')
    ax1.set_xlabel("Ri")
    ax1.set_ylabel(r"$\phi$")

    plt.legend()

    plt.savefig("stability_functions.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    make_comparison()
