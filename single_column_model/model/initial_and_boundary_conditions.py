# standard imports
import numpy as np
import fenics as fe

# project related imports
from single_column_model.utils import transform_values as tv


def def_initial_conditions(Q, mesh, params):
    z0 = params.z0  # roughness length in meter
    Nz = params.Nz  # number of point/ domain resolution
    H = params.H  # domain height in meters
    u_G = params.u_G  # u geostrophic wind
    initCondStr = params.init_path  # name of the file

    z = mesh.coordinates()

    if params.load_ini_cond:
        u_n = tv.convert_numpy_array_to_fenics_function(np.load(initCondStr + '_u.npy'), Q)
        v_n = tv.convert_numpy_array_to_fenics_function(np.load(initCondStr + '_v.npy'), Q)
        T_n = tv.convert_numpy_array_to_fenics_function(np.load(initCondStr + '_theta.npy'), Q)
        k_n = tv.convert_numpy_array_to_fenics_function(np.load(initCondStr + '_TKE.npy'), Q)

    else:
        u_n = tv.convert_numpy_array_to_fenics_function(initial_u_0(z, u_G, z0, params), Q)
        v_n = tv.convert_numpy_array_to_fenics_function(0 * np.ones(Nz), Q)
        T_n = tv.convert_numpy_array_to_fenics_function(initial_theta_0(z, params.T_ref, 200), Q)
        k_n = tv.convert_numpy_array_to_fenics_function(initial_k_0(z, u_G, z0, H) + 0.01, Q)

    return u_n, v_n, T_n, k_n


def def_boundary_conditions(fenics_params, params):
    z0 = params.z0  # roughness length in meter
    H = params.H  # domain height in meters
    u_G = params.u_G  # u geostrophic wind
    initCondStr = params.init_path  # name of the file

    load_ini_cond = params.load_ini_cond  # bool type; load existing initial condition

    V = fenics_params.W  # fenics variable; the vector function space

    ground = 'near(x[0],' + str(z0) + ',1E-6)'
    top = 'near(x[0],' + str(H) + ',1E-6)'

    if load_ini_cond:
        u_ini = np.load(initCondStr + '_u.npy')
        v_ini = np.load(initCondStr + '_v.npy')
        T_ini = np.load(initCondStr + '_theta.npy')
        k_ini = np.load(initCondStr + '_TKE.npy')

        u_D_low = fe.Expression('value', degree=0, value=u_ini[0])
        u_D_top = fe.Expression('value', degree=0, value=u_ini[-1])

        v_D_low = fe.Expression('value', degree=0, value=v_ini[0])
        T_D_low = fe.Expression('value', degree=0, value=T_ini[0])

        k_D_low = fe.Expression('value', degree=0, value=k_ini[0])
        k_D_top = fe.Expression('value', degree=0, value=k_ini[-1])

        # velocity u component
        bcu_ground = fe.DirichletBC(V.sub(0), u_D_low, ground)
        bcu_top = fe.DirichletBC(V.sub(0), u_D_top, top)

        # velocity v component
        bcv_ground = fe.DirichletBC(V.sub(1), v_D_low, ground)
        bcv_top = fe.DirichletBC(V.sub(1), 0.0, top)

        # Temperature
        bcT_ground = fe.DirichletBC(V.sub(2), T_D_low, ground)

        # TKE
        bck_ground = fe.DirichletBC(V.sub(3), k_D_low, ground)
        bck_top = fe.DirichletBC(V.sub(3), k_D_top, top)

        bc = [bcu_ground, bcv_ground, bcT_ground, bck_ground, bcv_top]

        Tg_n = T_ini[0]

    else:
        # Define velocity u component
        bc_u_ground = fe.DirichletBC(V.sub(0), 0.0, ground)

        # Define velocity v component
        bc_v_ground = fe.DirichletBC(V.sub(1), 0.0, ground)
        bc_v_top = fe.DirichletBC(V.sub(1), 0.0, top)

        # Define potential temperature
        theta_low = fe.Expression('value', degree=0, value=params.T_ref)
        bc_theta_ground = fe.DirichletBC(V.sub(2), theta_low, ground)

        # Define TKE
        k_D_low = fe.Expression('value', degree=0, value=initial_k_0(z0, u_G, z0, 200))
        bc_k_ground = fe.DirichletBC(V.sub(3), k_D_low, ground)

        # Combine boundary conditions
        bc = [bc_u_ground, bc_v_ground, bc_theta_ground, bc_k_ground, bc_v_top]

        #
        Tg_n = params.T_ref

    # writing out the fenics parameters
    fenics_params.bc = bc  # list of boundary conditions. Will be used in the FEM formulation
    fenics_params.theta_D_low = theta_low  # Temperature. Fenics expression is used to control the value within the main loop solution
    fenics_params.k_D_low = k_D_low  # TKE. Fenics expression is used to control the value within the main loop solution

    fenics_params.U_g = fe.Expression('value', degree=0,
                             value=params.u_G)  # Geostrophic wind; added here to control in the main loop
    fenics_params.V_g = fe.Constant(params.v_G)  # Geostrophic wind; added here to control in the main loop

    # writing out normal parameters
    params.Tg_n = Tg_n  # The value of the Temperature at the ground.

    return fenics_params, params


def initial_u_0(z, u_G, z0, params):
    """Calculate  initial profile for the wind velocity u.

        Args:
            z (fenics function): Mesh coordinates.
            u_G (float): Zonal geostrophic wind speed.
            z0 (float): Surface roughness length.
            params (class): Parameters from dataclass. For more details see parameters.py

        Returns:
            (list?): initial profile for u
    """
    # Set tuning parameter
    c_f = 4 * 10 ** (-3)
    # Calculate initial friction velocity
    u_star_ini = np.sqrt(0.5 * c_f * u_G ** 2)
    return u_star_ini / params.kappa * np.log(z / z0)


def initial_theta_0(z, theta_initial, cut_height):
    """Calculate  initial profile for the temperature theta.

    Args:
        z (fenics function): Mesh coordinates.
        theta_initial (float): Initial potential temperature.
        cut_height (float): Height at which the temperature depends on the laps rate and the height of the grid point.

    Returns:
        (list?): initial profile for theta
    """
    # Set atmospheric laps rate [K/m]
    gamma = 0.01
    # Set initial theta above reference height
    theta_0 = (z - cut_height) * gamma + theta_initial
    # Find index of value where z - cut_height = 0
    index_cut_height = int(np.abs(z - cut_height).argmin())
    # Set initial theta below reference height
    theta_0[0:index_cut_height + 1] = theta_initial
    return theta_0


def initial_k_0(z, u_G, z0, H, k_at_H=0.0):
    """Calculate  initial profile for TKE. See Parente, A., C. Gorlé, J. van Beeck, and C. Benocci, 2011: A
    Comprehensive Modelling Approach for the Neutral Atmospheric Boundary Layer: Consistent Inflow Conditions, Wall
    Function and Turbulence Model. Boundary-Layer Meteorol, 140, 411–428, https://doi.org/10.1007/s10546-011-9621-5.

    Args:
        params (class): Parameters from dataclass. For more details see parameters.py.
        z (fenics function): Mesh coordinates.
        u_G (float): Zonal geostrophic wind speed.
        z0 (float): Surface roughness length.
        H (float): Domain height.
        k_at_H (float): TKE for t=0 and z=H.

    Returns:
        (list?): initial profile for TKE
    """
    # Set tuning parameter
    c_f = 4*10**(-3)
    # Calculate initial friction velocity
    u_star_ini = np.sqrt(0.5 * c_f * u_G ** 2)
    # TKE for t=0 and z=z0
    k_at_z0 = u_star_ini ** 2 / np.sqrt(0.087)
    # Solve system of equations
    func = lambda z, a, b: a * np.log(z+z0) + b
    a = (k_at_H - k_at_z0) / (np.log(H) - np.log(z0))
    b = k_at_z0 - a * np.log(z0)
    return func(z, a, b)
