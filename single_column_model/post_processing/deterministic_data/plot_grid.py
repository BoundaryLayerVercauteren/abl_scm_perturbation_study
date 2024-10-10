import numpy as np
import matplotlib.pyplot as plt

Nz = 100  # number of point/ domain resolution
z0 = 0.044  # roughness length in meter
H = 300.0

lb = z0 ** (1 / 3)
rb = H ** (1 / 3)
space = np.linspace(lb, rb, Nz)
grid = space**3

print(grid[:-1] - grid[1:])
exit()
x = np.repeat(0, len(grid))

fig, ax = plt.subplots(1, 1, figsize=(5, 10), constrained_layout=True)

ax.plot(x, grid)
ax.scatter(x, grid, marker="s", s=3, c="red")
ax.set_ylabel("z [h]")
ax.grid()

plt.savefig("power_grid.png", bbox_inches="tight", dpi=300)
