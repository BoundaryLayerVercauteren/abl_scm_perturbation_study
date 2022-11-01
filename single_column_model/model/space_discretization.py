# standard imports
import matplotlib.pyplot as plt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py

# project related imports
from single_column_model.model import utility_functions as ut


# -----------------------------------------------------------------------------
def create_grid(params, grid_type, show=False):
    z0 = params.z0  # roughness length
    H = params.H  # domain height in meters
    Nz = params.Nz  # number of point/ domain resolution

    grid = []

    if grid_type == 'power':
        grid = power_grid(z0, H, Nz)

    if grid_type == 'log':
        grid = log_grid(z0, H, Nz)

    if grid_type == 'log_lin':
        grid = log_lin_grid(z0, H, Nz)

    if grid_type == 'linear':
        grid = lin_grid(z0, H, Nz)

    if grid_type == 'for_tests':
        grid = grid_for_tests(z0, H, Nz)

    if show:
        plt.figure()
        plt.title('Grid spacing')
        plt.plot(grid, 'o')
        plt.xlabel('n points')
        plt.ylabel('space')
        plt.show()

    if grid_type == 'for_tests':
        grid.shape = (Nz + 1, 1)
        # define fenics mesh object
        mesh = IntervalMesh(Nz, z0, H)
    else:
        grid.shape = (Nz, 1)
        # define fenics mesh object
        mesh = IntervalMesh(Nz - 1, z0, H)

        # set new mesh
    X = mesh.coordinates()
    X[:] = grid

    return mesh, params


# -----------------------------------------------------------------------------


def power_grid(z0, H, Nz):
    lb = z0 ** (1 / 3)
    rb = H ** (1 / 3)
    space = np.linspace(lb, rb, Nz)
    return space ** 3


def log_grid(z0, H, Nz):
    return np.logspace(np.log10(z0), np.log10(H), Nz)


def log_lin_grid(z0, H, Nz):
    b0 = 2.5
    return (np.logspace(np.log10(z0), np.log10(H), Nz) + np.linspace(z0, H, Nz)) / 2


def lin_grid(z0, H, Nz):
    return np.linspace(z0, H, Nz)


def grid_for_tests(z0, H, Nz):
    dz = (H - z0) / (Nz)
    return np.arange(z0, H + dz, dz)
