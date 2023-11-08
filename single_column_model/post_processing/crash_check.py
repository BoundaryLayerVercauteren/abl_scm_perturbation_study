import h5py
import matplotlib.pyplot as plt
import numpy as np

sim_dir = 'single_column_model/solution/long_tail/perturbed_gaussian/500_10/neg_u/simulations/'
perturb_list = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
sol_files = [f'{sim_dir}solution_uG_1.8_perturbstr_{r}_sim_0.h5' for r in perturb_list]

# Find crash time point
with h5py.File(sol_files[-1], "r+") as file:
    crashed_ri = file["Ri"][:]

idx_crashed = np.isnan(crashed_ri).argmax(axis=1) - 1

# print(np.min(crashed_ri[:,idx_crashed]))

min_ri = []
max_ri = []
min_km = []
max_km = []
min_kh = []
max_kh = []
min_tke = []
max_tke = []

for idx, sol_file in enumerate(sol_files):
    with h5py.File(sol_file, "r+") as file:
        ri = file["Ri"][:]
        km = file["Km"][:]
        kh = file["Kh"][:]
        tke = file["TKE"][:]

    plt.figure(figsize=(5, 5))
    for row in np.arange(0, 100):
        plt.plot(ri[row, :])
    plt.savefig(f'ri_val_{idx}.png')

    plt.figure(figsize=(5, 5))
    for row in np.arange(0, 100):
        plt.plot(km[row, :])
    plt.savefig(f'km_val_{idx}.png')

    plt.figure(figsize=(5, 5))
    for row in np.arange(0, 100):
        plt.plot(kh[row, :])
    plt.savefig(f'kh_val_{idx}.png')

    plt.figure(figsize=(5, 5))
    for row in np.arange(0, 100):
        plt.plot(tke[row, :])
    plt.savefig(f'tke_val_{idx}.png')

    min_ri.append(np.nanmin(ri))
    max_ri.append(np.nanmax(ri))

    min_km.append(np.nanmin(km))
    max_km.append(np.nanmax(km))

    min_kh.append(np.nanmin(kh))
    max_kh.append(np.nanmax(kh))

    min_tke.append(np.nanmin(tke))
    max_tke.append(np.nanmax(tke))

print(min_ri, '\n', max_ri, '\n\n', min_km, '\n', max_km, '\n\n', min_kh, '\n', max_kh, '\n\n', min_tke, '\n', max_tke)
