# standard imports
import fenics as fe


def l_m(fparams, params):
    return (params.kappa * fparams.x[0]) / (f_m(fparams, params) + (params.kappa * fparams.x[0]) / lambbda(fparams.U_g, params.f_c))


# Cuxart 2006 table 3
def K_m(fparams, params):
    return params.alpha * l_m(fparams, params) * fe.sqrt(fparams.k + 1E-16)


# Cuxart 2006 table 3 
def K_h(fparams, params):
    return K_m(fparams, params) / params.Pr_t


# Rodrigo 2013 Eq. 27, The TKE Generation term due to wind shear and buoyancy.
def G_E(fparams, params):
    theta = fparams.theta
    u = fparams.u
    v = fparams.v

    g = params.g
    T_ref = params.T_ref

    return K_m(fparams, params) * S(u, v) - g / (T_ref) * K_h(fparams, params) * theta.dx(0)


# Rodrigo 2013 Eq. 29, The turbulent dissipation rate eps
def eps(fparams, params):
    return params.alpha_e * (fparams.k) ** (3 / 2) * (1 / l_m(fparams, params) + 1E-16)


# Sorbjan 2012 after Eq. 3b. Since we did not take the sqrt(S) and sqrt(N) we
# do not compute N**2 / S**2 but N / S. We add the EPS, so we do not get division
# by zero.
def Ri(fparams, params):
    calc_Ri = abs(N(fparams.theta, params) / (S(fparams.u, fparams.v) + 1E-16))
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


def lambbda(U_g, f_c):
    return 3.7 * 1e-4 * U_g / f_c


# Sorbjan 2012 Eq. 5a and 5b. Empirical stability functions. They correct the
# turbulent mixing depending on the local stability.
def f_m(fparams, params):
    return 1.0 + 12.0 * Ri(fparams, params)


# as above. The heat is different from momentum
def f_h(fparams, params):
    return 1.0 + 12.0 * Ri(fparams, params)


def weak_formulation(fparams, params, u_n, v_n, T_n, k_n):
    u, v, theta, k, w_u, w_v, w_T, w_k, x, U_g, V_g = unroll(fparams)

    Dt = fe.Constant(params.dt)
    Fc = fe.Constant(params.f_c)
    tau = fe.Constant(params.tau)

    # Define variational problem
    # ---------- Velocity--u-comp------------------
    F_u = - Fc * (v - V_g) * w_u * fe.dx \
          + K_m(fparams, params) * fe.dot(u.dx(0), w_u.dx(0)) * fe.dx \
          + ((u - u_n) / Dt) * w_u * fe.dx \
          + (u - U_g) / tau * w_u * fe.dx

    # ---------- Velocity--v-comp------------------
    F_v = + Fc * (u - U_g) * w_v * fe.dx \
          + K_m(fparams, params) * fe.dot(v.dx(0), w_v.dx(0)) * fe.dx \
          + ((v - v_n) / Dt) * w_v * fe.dx \
          + (v - V_g) / tau * w_v * fe.dx

    # --------------Temperature--------------------
    F_theta = + K_h(fparams, params) * fe.dot(theta.dx(0), w_T.dx(0)) * fe.dx \
              - K_h(fparams, params) * params.gamma * w_T * fe.ds \
              + ((theta - T_n) / Dt) * w_T * fe.dx

    # ------------------TKE------------------------
    F_k = + K_m(fparams, params) * fe.dot(k.dx(0), w_k.dx(0)) * fe.dx \
          - G_E(fparams, params) * w_k * fe.dx \
          + eps(fparams, params) * w_k * fe.dx \
          + ((k - k_n) / Dt) * w_k * fe.dx

    F = F_u + F_v + F_theta + F_k

    return F


def setup_fenics_variables(fparams, mesh):
    fparams.W = fe.VectorFunctionSpace(mesh, 'CG', 1, dim=4)

    # Define test functions
    fparams.w_u, fparams.w_v, fparams.w_T, fparams.w_k = fe.TestFunctions(fparams.W)

    # Split system functions to access components
    fparams.uvTk = fe.Function(fparams.W)
    fparams.u, fparams.v, fparams.theta, fparams.k = fe.split(fparams.uvTk)

    # height "z"
    fparams.x = fe.SpatialCoordinate(mesh)
    fparams.z = mesh.coordinates()  # Grid of the simulation domain.

    # Function space for projection. For writing out variables
    fparams.Q = fe.FunctionSpace(mesh, "CG", 1)

    return fparams


def prepare_fenics_solver(fparams, F):
    fe.set_log_level(fe.LogLevel.WARNING)  # supress fenics output

    J = fe.derivative(F, fparams.uvTk)

    problem = fe.NonlinearVariationalProblem(F, fparams.uvTk, fparams.bc, J)

    solver = fe.NonlinearVariationalSolver(problem)
    # fe.info(solver.parameters, True)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    solver.parameters["newton_solver"]['absolute_tolerance'] = 1E-6
    solver.parameters["newton_solver"]['relative_tolerance'] = 1E-6

    return solver


def unroll(fparams):
    u = fparams.u
    v = fparams.v
    theta = fparams.theta
    k = fparams.k

    w_u = fparams.w_u
    w_v = fparams.w_v
    w_T = fparams.w_T
    w_k = fparams.w_k

    x = fparams.x

    U_g = fparams.U_g
    V_g = fparams.V_g

    return u, v, theta, k, w_u, w_v, w_T, w_k, x, U_g, V_g
