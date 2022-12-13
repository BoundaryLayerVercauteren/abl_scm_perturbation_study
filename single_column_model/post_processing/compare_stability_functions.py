import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cmcrameri.cm as cram  # Package for plot colors

plt.style.use('science')

# set font sizes for plots
SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def define_vandewiel_short_tail_exp_stab_function(Ri, alpha=5):
    return np.exp(-2 * alpha * Ri - (alpha * Ri) ** 2)


def define_vandewiel_short_tail_stab_function(Ri, alpha=5):
    return (1 - alpha * Ri) ** 2


def define_slavas_long_tail_stab_function(Ri):
    return 1 / (1 + 12 * Ri)


def make_comparison():
    richardson_num = np.linspace(0.001, 2, 100)

    vec_vandewiel_short_tail_exp_stab_func = np.vectorize(define_vandewiel_short_tail_exp_stab_function)
    # vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    vec_slavas_long_tail_stab_func = np.vectorize(define_slavas_long_tail_stab_function)

    # Create plot
    color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors
    markers = ['v', '*', '^', 's', 'p', '.']

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    # ax1.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail',
    #          color=color[0], marker=markers[0], markevery=10)
    ax1.plot(richardson_num, vec_slavas_long_tail_stab_func(richardson_num), label=r'$(1 + 12  Ri)^{-1}$',
             color=color[1], marker=markers[1], markevery=10)
    ax1.plot(richardson_num, vec_vandewiel_short_tail_exp_stab_func(richardson_num), label=r'$\exp(-2  \alpha  Ri - (\alpha  Ri) ^ 2)$',
             color=color[2], marker=markers[2], markevery=10)

    ax1.set_xlabel('Ri')
    ax1.set_ylabel(r'$f$')

    plt.legend()

    plt.savefig('stability_functions.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    make_comparison()
