"""Script to handle fenics functions/ variables."""

import fenics as fe
import numpy as np


def interpolate_fenics_function_to_numpy_array(fenics_function, vector_space):
    """Interpolate a fenics function to turn into a numpy array
    :param fenics_function: function to be transformed
    :param vector_space: function space
    """
    return np.flipud(fe.interpolate(fenics_function, vector_space).vector().get_local())


def project_fenics_function_to_numpy_array(fenics_function, vector_space):
    """Project a fenics function to turn into a numpy array
    :param fenics_function: function to be transformed
    :param vector_space: function space
    """
    return np.flipud(fe.project(fenics_function, vector_space).vector().get_local())


def convert_numpy_array_to_fenics_function(values, vector_space):
    """Pass values from numpy array to fenics function
    :param values: values to passed
    :param vector_space: function space for fenics function
    """
    # Create help function for projection
    t1 = fe.Function(vector_space)

    # Create vector with values
    t1.vector().set_local(np.flipud(values))

    return fe.project(t1, vector_space)
