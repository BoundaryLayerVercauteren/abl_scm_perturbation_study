import scienceplots
import matplotlib.pyplot as plt

def set_style_of_plots(figsize):
    plt.style.use("science")

    # Set font sizes for plots
    if figsize == (10,5) or figsize == (5,5):
        SMALL_SIZE = 11
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 15
    elif figsize == (10,10):
        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        BIGGER_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title