# standard imports
import fenics as fe


def define_short_tail_stability_function(fenics_params, params):
    """See Delage (1997)."""
    return 1.0 + 12.0 * Ri(fenics_params, params)


def define_long_tail_stability_function(fenics_params, params):
    """See Delage (1997)."""
    return 1.0 + 4.7 * Ri(fenics_params, params)


def sigmoid(fenics_params, params, k=0.1):
    return 1 / (1 + fe.exp(-k * (fenics_params.x[0] - params.z_l)))


def f_m(fenics_params, params):
    """Stability function for momentum."""
    if params.stab_func_type == "long_tail":
        fm = define_long_tail_stability_function(fenics_params, params)
    elif params.stab_func_type == "short_tail":
        fm = define_short_tail_stability_function(fenics_params, params)

    if params.perturbation_param == "stab_func":
        fm = fm * sigmoid(fenics_params, params) + fenics_params.f_ms * (
            1 - sigmoid(fenics_params, params)
        )
    return fm


def lambda_param(u_G, f_c):
    """Parameter to restrain the largest turbulent eddies. See Rodrigo et al. (2013)."""
    return 2.7 * 1e-4 * u_G / f_c


def l_m(fenics_params, params):
    """Turbulent mixing length. See Rodrigo et al. (2013) and Delage (1974)."""
    return (params.kappa * fenics_params.x[0]) / (
        f_m(fenics_params, params)
        + params.kappa
        * fenics_params.x[0]
        / lambda_param(fenics_params.U_g, params.f_c)
    )


def f_h(fenics_params, params):
    """Stability function for heat."""
    return f_m(fenics_params, params)


def K_m(fenics_params, params):
    """Momentum diffusion equation. See Rodrigo et al. (2013)."""
    return params.alpha * l_m(fenics_params, params) * fe.sqrt(fenics_params.k + 1e-16)


def K_h(fenics_params, params):
    """Heat diffusion equation. See Rodrigo et al. (2013)."""
    return K_m(fenics_params, params) / params.Pr_t


def Ri(fenics_params, params):
    """Gradient Richardson number."""
    calc_Ri = abs(
        (params.g / params.theta_ref * fenics_params.theta.dx(0))
        / (fenics_params.u.dx(0) ** 2 + fenics_params.v.dx(0) ** 2 + 1e-16)
    )
    # Limit Ri to max value of 10 (for solver stability)
    return fe.conditional(fe.gt(calc_Ri, 10.0), 10.0, calc_Ri)


def weak_formulation(fenics_params, params, u_n, v_n, T_n, k_n):
    u, v, theta, k, u_test, v_test, theta_test, k_test, x, U_g, V_g = unroll(
        fenics_params
    )

    Dt = fe.Constant(params.dt)
    Fc = fe.Constant(params.f_c)
    tau = fe.Constant(params.tau)

    # Define variational problem
    # ---------- Velocity--u-comp------------------
    F_u = (
        -Fc * (v - V_g) * u_test * fe.dx
        + K_m(fenics_params, params) * fe.dot(u.dx(0), u_test.dx(0)) * fe.dx
        + ((u - u_n) / Dt) * u_test * fe.dx
        + (u - U_g) / tau * u_test * fe.dx
    )

    # ---------- Velocity--v-comp------------------
    F_v = (
        +Fc * (u - U_g) * v_test * fe.dx
        + K_m(fenics_params, params) * fe.dot(v.dx(0), v_test.dx(0)) * fe.dx
        + ((v - v_n) / Dt) * v_test * fe.dx
        + (v - V_g) / tau * v_test * fe.dx
    )

    # --------------Temperature--------------------
    F_theta = (
        +K_h(fenics_params, params) * fe.dot(theta.dx(0), theta_test.dx(0)) * fe.dx
        - K_h(fenics_params, params) * params.gamma * theta_test * fe.ds
        + ((theta - T_n) / Dt) * theta_test * fe.dx
    )

    # ------------------TKE------------------------
    F_k = (
        +K_m(fenics_params, params) * fe.dot(k.dx(0), k_test.dx(0)) * fe.dx
        - K_m(fenics_params, params) * (u.dx(0) ** 2 + v.dx(0) ** 2) * k_test * fe.dx
        + (
            params.g
            / params.theta_ref
            * K_h(fenics_params, params)
            * theta.dx(0)
            * k_test
            * fe.dx
        )
        + (
            (params.alpha_e * fenics_params.k) ** (3 / 2)
            * (1 / l_m(fenics_params, params) + 1e-16)
            * k_test
            * fe.dx
        )
        + ((k - k_n) / Dt) * k_test * fe.dx
    )

    F = F_u + F_v + F_theta + F_k

    return F


def setup_fenics_variables(fenics_params, mesh):
    fenics_params.W = fe.VectorFunctionSpace(mesh, "CG", 1, dim=4)

    # Define test functions
    (
        fenics_params.u_test,
        fenics_params.v_test,
        fenics_params.theta_test,
        fenics_params.k_test,
    ) = fe.TestFunctions(fenics_params.W)

    # Split system functions to access components
    fenics_params.uvTk = fe.Function(fenics_params.W)
    fenics_params.u, fenics_params.v, fenics_params.theta, fenics_params.k = fe.split(
        fenics_params.uvTk
    )

    # height "z"
    fenics_params.x = fe.SpatialCoordinate(mesh)
    fenics_params.z = mesh.coordinates()  # Grid of the simulation domain.

    # Function space for projection. For writing out variables
    fenics_params.Q = fe.FunctionSpace(mesh, "CG", 1)

    return fenics_params


def prepare_fenics_solver(fenics_params, F):
    fe.set_log_level(fe.LogLevel.WARNING)  # suppress fenics output

    J = fe.derivative(F, fenics_params.uvTk)

    problem = fe.NonlinearVariationalProblem(F, fenics_params.uvTk, fenics_params.bc, J)

    solver = fe.NonlinearVariationalSolver(problem)
    # fe.info(solver.parameters, True)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-6
    solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6

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
