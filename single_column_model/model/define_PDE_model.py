# standard imports
import fenics as fe
import numpy as np


def lambda_param(u_G, f_c):
    """Parameter to restrain (in neutral stratification) the largest turbulent eddies. See Rodrigo, J. S., and P. S.
    Anderson, 2013: Investigation of the Stable Atmospheric Boundary Layer at Halley Antarctica. Boundary-Layer
    Meteorol, 148, 517–539, https://doi.org/10.1007/s10546-013-9831-0.

    Args:
        u_G (float): Zonal geostrophic wind speed.
        f_c (float): Coriolis parameter.

    Returns:
        (float): lambda parameter
    """
    return 2.7 * 1e-4 * u_G / f_c


def l_m(fenics_params, params):
    """ Turbulent mixing length. See Cuxart, J., and Coauthors, 2006: Single-Column Model Intercomparison for a Stably
    Stratified Atmospheric Boundary Layer. Boundary-Layer Meteorol, 118, 273–303,
    https://doi.org/10.1007/s10546-005-3780-1.


    Args:
        params (class): Parameters from dataclass. For more details see parameters.py
        fenics_params (class): Parameter class for fenics parameters (i.e. mesh, function space)

    Returns:
        (float): turbulent mixing length
    """
    lambda_val = lambda_param(fenics_params.U_g, params.f_c)
    return (params.kappa * fenics_params.x[0] * lambda_val) / (params.kappa * fenics_params.x[0] + lambda_val)


# Cuxart 2006 table 3
def K_m(fenics_params, params):
    return params.alpha * l_m(fenics_params, params) * fe.sqrt(fenics_params.k + 1E-16) * f_m(fenics_params, params)


# Cuxart 2006 table 3 
def K_h(fenics_params, params):
    return K_m(fenics_params, params) / params.Pr_t


# Rodrigo 2013 Eq. 27, The TKE Generation term due to wind shear and buoyancy.
def G_E(fenics_params, params):
    theta = fenics_params.theta
    u = fenics_params.u
    v = fenics_params.v

    g = params.g
    T_ref = params.T_ref

    return K_m(fenics_params, params) * S(u, v) - g / (T_ref) * K_h(fenics_params, params) * theta.dx(0)


# Rodrigo 2013 Eq. 29, The turbulent dissipation rate eps
def eps(fenics_params, params):
    return params.alpha_e * (fenics_params.k) ** (3 / 2) * (1 / l_m(fenics_params, params) + 1E-16)


# Sorbjan 2012 after Eq. 3b. Since we did not take the sqrt(S) and sqrt(N) we
# do not compute N**2 / S**2 but N / S. We add the EPS, so we do not get division
# by zero.
def Ri(fenics_params, params):
    calc_Ri = abs(N(fenics_params.theta, params) / (S(fenics_params.u, fenics_params.v) + 1E-16))
    # limit the Ri to max value of 10 (for solver stability)
    return fe.conditional(fe.gt(calc_Ri, 10.0), 10.0, calc_Ri)


# Sorbjan 2012 Eq. 3b
# Gradients of the wind. For computing the Ri Number and the eddy diffusivities.
# We take the sqrt(S) later in the definition of the eddy diffusivities. 
def S(u, v):
    return u.dx(0) ** 2 + v.dx(0) ** 2


# Sorbjan 2012 Eq. 3b
# Brunt-Vaisala frequency. For computing the Ri Number. We do not take the
# sqrt(N) to allow for negativ values.
def N(theta, params):
    return params.beta * theta.dx(0)


def define_long_tail_stability_function(fenics_params, params):
    """Cuxart, J., and Coauthors, 2006: Single-Column Model Intercomparison for a Stably Stratified Atmospheric Boundary
    Layer. Boundary-Layer Meteorol, 118, 273–303, https://doi.org/10.1007/s10546-005-3780-1.
    """
    return (1.0 + 12.0 * Ri(fenics_params, params))**(-1)


def define_short_tail_stability_function(fenics_params, params, alpha=5):
    """Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature inversions: A
    conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073, https://doi.org/10.1175/JAS-D-16-0180.1.
    """
    Ri_val = Ri(fenics_params, params)
    return np.exp(-2 * alpha * Ri_val - (alpha * Ri_val) ** 2)


def f_m(fenics_params, params):
    """Stability function for momentum."""
    if params.stab_func_type == 'long_tail':
        fm = define_long_tail_stability_function(fenics_params, params)
    elif params.stab_func_type == 'short_tail':
        fm = define_short_tail_stability_function(fenics_params, params)
    return fm


def f_h(fenics_params, params):
    """Stability function for heat."""
    return f_m(fenics_params, params)


def weak_formulation(fenics_params, params, u_n, v_n, T_n, k_n):
    u, v, theta, k, u_test, v_test, theta_test, k_test, x, U_g, V_g = unroll(fenics_params)

    Dt = fe.Constant(params.dt)
    Fc = fe.Constant(params.f_c)
    tau = fe.Constant(params.tau)

    # Define variational problem
    # ---------- Velocity--u-comp------------------
    F_u = - Fc * (v - V_g) * u_test * fe.dx \
          + K_m(fenics_params, params) * fe.dot(u.dx(0), u_test.dx(0)) * fe.dx \
          + ((u - u_n) / Dt) * u_test * fe.dx \
          + (u - U_g) / tau * u_test * fe.dx

    # ---------- Velocity--v-comp------------------
    F_v = + Fc * (u - U_g) * v_test * fe.dx \
          + K_m(fenics_params, params) * fe.dot(v.dx(0), v_test.dx(0)) * fe.dx \
          + ((v - v_n) / Dt) * v_test * fe.dx \
          + (v - V_g) / tau * v_test * fe.dx

    # --------------Temperature--------------------
    F_theta = + K_h(fenics_params, params) * fe.dot(theta.dx(0), theta_test.dx(0)) * fe.dx \
              - K_h(fenics_params, params) * params.gamma * theta_test * fe.ds \
              + ((theta - T_n) / Dt) * theta_test * fe.dx

    # ------------------TKE------------------------
    F_k = + K_m(fenics_params, params) * fe.dot(k.dx(0), k_test.dx(0)) * fe.dx \
          - G_E(fenics_params, params) * k_test * fe.dx \
          + eps(fenics_params, params) * k_test * fe.dx \
          + ((k - k_n) / Dt) * k_test * fe.dx

    F = F_u + F_v + F_theta + F_k

    return F


def setup_fenics_variables(fenics_params, mesh):
    fenics_params.W = fe.VectorFunctionSpace(mesh, 'CG', 1, dim=4)

    # Define test functions
    fenics_params.u_test, fenics_params.v_test, fenics_params.theta_test, fenics_params.k_test = fe.TestFunctions(
        fenics_params.W)

    # Split system functions to access components
    fenics_params.uvTk = fe.Function(fenics_params.W)
    fenics_params.u, fenics_params.v, fenics_params.theta, fenics_params.k = fe.split(fenics_params.uvTk)

    # height "z"
    fenics_params.x = fe.SpatialCoordinate(mesh)
    fenics_params.z = mesh.coordinates()  # Grid of the simulation domain.

    # Function space for projection. For writing out variables
    fenics_params.Q = fe.FunctionSpace(mesh, "CG", 1)

    return fenics_params


def prepare_fenics_solver(fenics_params, F):
    fe.set_log_level(fe.LogLevel.WARNING)  # supress fenics output

    J = fe.derivative(F, fenics_params.uvTk)

    problem = fe.NonlinearVariationalProblem(F, fenics_params.uvTk, fenics_params.bc, J)

    solver = fe.NonlinearVariationalSolver(problem)
    # fe.info(solver.parameters, True)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    solver.parameters["newton_solver"]['absolute_tolerance'] = 1E-6
    solver.parameters["newton_solver"]['relative_tolerance'] = 1E-6

    return solver


def unroll(fenics_params):
    u = fenics_params.u
    v = fenics_params.v
    theta = fenics_params.theta
    k = fenics_params.k

    u_test = fenics_params.u_test
    v_test = fenics_params.v_test
    theta_test = fenics_params.theta_test
    k_test = fenics_params.k_test

    x = fenics_params.x

    U_g = fenics_params.U_g
    V_g = fenics_params.V_g

    return u, v, theta, k, u_test, v_test, theta_test, k_test, x, U_g, V_g
