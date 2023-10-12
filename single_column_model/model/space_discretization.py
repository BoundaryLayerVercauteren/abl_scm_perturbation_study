# coding=utf-8
# !/usr/bin/env python

import fenics as fe
import numpy as np


# -----------------------------------------------------------------------------
def create_grid(params, grid_type):
    z0 = params.z0  # roughness length
    H = params.H  # domain height in meters
    Nz = params.Nz  # number of point/ domain resolution

    grid = []

    if grid_type == "power":
        grid = power_grid(z0, H, Nz)
    elif grid_type == "log":
        grid = log_grid(z0, H, Nz)
    elif grid_type == "log_lin":
        grid = log_lin_grid(z0, H, Nz)
    elif grid_type == "linear":
        grid = lin_grid(z0, H, Nz)
    elif grid_type == "for_tests":
        grid = grid_for_tests(z0, H, Nz)

    if grid_type == "for_tests":
        grid.shape = (Nz + 1, 1)
        # Define fenics mesh object
        mesh = fe.IntervalMesh(Nz, z0, H)
    else:
        grid.shape = (Nz, 1)
        # Define fenics mesh object
        mesh = fe.IntervalMesh(Nz - 1, z0, H)

    # Set mesh coordinates to the ones given by the grid
    mesh.coordinates()[:] = grid

    return mesh


# -----------------------------------------------------------------------------


def power_grid(z0, H, Nz):
    lb = z0 ** (1 / 3)
    rb = H ** (1 / 3)
    space = np.linspace(lb, rb, Nz)
    return space ** 3


def log_grid(z0, H, Nz):
    return np.logspace(np.log10(z0), np.log10(H), Nz)


def log_lin_grid(z0, H, Nz):
    return (np.logspace(np.log10(z0), np.log10(H), Nz) + np.linspace(z0, H, Nz)) / 2


def lin_grid(z0, H, Nz):
    return np.linspace(z0, H, Nz)


def grid_for_tests(z0, H, Nz):
    dz = (H - z0) / (Nz)
    return np.arange(z0, H + dz, dz)
