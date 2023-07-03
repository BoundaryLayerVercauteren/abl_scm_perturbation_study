import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from cmcrameri import cm
import scienceplots

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


def make_3d_plot(output, params, fparams, file_spec=''):
    # Create grid for plotting
    t = np.linspace(0, params.T_end_h, params.save_num_steps)
    z = fparams.z
    X, Y = np.meshgrid(t, z)

    values = [output.U_save, output.V_save, output.T_save, output.k_save]
    title = ['U [m/s]', 'V [m/s]', r"$\theta$ [K]", r'TKE [$m^2/s^2$]']
    file_names = ['u_over_time_z_' + file_spec + '.png',
                  'v_over_time_z_' + file_spec + '.png',
                  'theta_over_time_z_' + file_spec + '.png',
                  'tke_over_time_z_' + file_spec + '.png']
    colours = [cm.davos, cm.davos, cm.lajolla, cm.tokyo]
    # color_range = [(0.0, 5.0), (0.0, 1.5), (298.0, 302.0)]

    # Create plot for all variables
    for index in range(len(title)):
        plt.figure(figsize=(5, 5))
        plt.rc('font', **{'size': '11'})
        plt.pcolor(X, Y, values[index],
                   cmap=colours[index])  # , vmin=color_range[index][0], vmax=color_range[index][1])
        plt.xlabel('time [h]')
        plt.ylabel('z [m]')
        plt.colorbar()
        plt.title(title[index])
        plt.savefig(str(output.solution_directory) + file_names[index], bbox_inches='tight')

        # To clear memory
        plt.cla()  # Clear the current axes.
        plt.clf()  # Clear the current figure.
        plt.close('all')  # Closes all the figure windows.