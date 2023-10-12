import h5py
import matplotlib.pyplot as plt

data_path = 'FLOSS/FLOSS2filterOnlyNights.h5'

with h5py.File(data_path, "r+") as file:
    theta_1m = file["/1height1m/T_night"][:].flatten()
    theta_20m = file["/6height20m/T_night"][:].flatten()
    theta_30m = file["/7height30m/T_night"][:].flatten()

    u_20m = file["/6height20m/u_mean_night"][:].flatten()
    u_30m = file["/7height30m/u_mean_night"][:].flatten()

delta_theta_20m = theta_20m - theta_1m
delta_theta_30m = theta_30m - theta_1m

# fig, ax = plt.subplots(2, 1, figsize=(10, 5))
# ax[0].plot(delta_theta_20m, color='blue', label='20m')
# ax[0].set_title('20m')
# ax[0].set_ylabel(r'$\Delta \theta$ [K]')
#
# ax[1].plot(delta_theta_30m, color='red', label='30m')
# ax[1].set_title('30m')
# ax[1].set_ylabel(r'$\Delta \theta$ [K]')
#
# plt.tight_layout()
# fig.savefig('FLOSS/delta_theta_over_idx.png', dpi=300)
#
# # Clear memory
# plt.cla()  # Clear the current axes.
# plt.clf()  # Clear the current figure.
# plt.close("all")  # Closes all the figure windows.

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(u_20m, delta_theta_20m, color='blue', label='20m', s=0.5)
ax[0].set_title('20m')
ax[0].set_ylabel(r'$\Delta \theta$ [K]')
ax[0].set_xlabel(r'$u_{20m}$ [m/s]')

ax[1].scatter(u_30m, delta_theta_30m, color='red', label='30m', s=0.5)
ax[1].set_title('30m')
ax[1].set_ylabel(r'$\Delta \theta$ [K]')
ax[1].set_xlabel(r'$u_{30m}$ [m/s]')

plt.tight_layout()
fig.savefig('FLOSS/delta_theta_over_u.png', dpi=300)

# Clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close("all")  # Closes all the figure windows.