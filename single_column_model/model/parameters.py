# standard imports
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np


@dataclass_json
@dataclass
class Parameters:
    save_ini_cond: bool = False  # save simulations solution as initial condition
    load_ini_cond: bool = False  # load existing initial condition

    T_end_h: float = 10  # hour
    T_end: float = T_end_h * 3600  # seconds
    dt: float = 10  # seconds
    num_steps: int = int(T_end / dt)  # number of time steps

    stab_func_type: str = 'long_tail'  # type of stability function

    perturbation_param: str = 'none'  # specify to which equation a perturbation is added
    perturbation_type: str = 'gauss_process'  # type of perturbation to be added
    perturbation_strength: float = 0.03  # strength of perturbation
    perturbation_start: int = int(0.5 * 3600 / dt)  # start time of perturbation
    perturbation_length: int = num_steps - perturbation_start + 1  # length of perturbation

    num_simulation: int = 1
    num_proc: int = 5

    # file name for initial conditions
    init_cond_path: str = 'steady_state_'#'T_end_' + str(T_end_h) + 'h_'

    # time steps to save
    save_dt: float = 60  # in seconds, needs to be bigger or equal to dt
    save_dt_sim: int = int(save_dt / dt)
    save_num_steps: int = int(T_end / save_dt)

    Nz: int = 100  # number of point/ domain resolution
    z0: float = 0.044  # roughness length in meter
    z0h: float = z0 * 0.1  # roughness length for heat in meter
    H: float = 300.0  # domain height in meters  ! should be H > z_l * s_dom_ext

    omega: float = (2 * np.pi) / (24 * 60 * 60)  # angular earth velocity
    theta_m: float = 290  # restoring temperature of peat soil [K]
    T_ref: float = 300  # reference potential temperature [K]
    theta_A: float = 300  # potential temperature at the boundary layer top [K]
    rho: float = 1.225  # air density kg/m**3 at 15 C
    C_p: float = 1005  # specific heat capacity at constant pressure of air
    C_g: float = 0.95 * (1.45 * 3.58 * 1e+6 / 2 / omega) ** 0.5  # heat capacity of ground per unit area
    sig: float = 5.669e-8  # non-dimensional Stefan-Boltzmann constant
    Q_c: float = 0.0  # the cloud fraction
    Q_a: float = 0.003  # specific humidity [g kg^-1]
    Tg_n: float = 300  # temperature initial value at the ground [K].
    R_n: float = -30 # -30 net radiations for surface energy balance
    k_m: float = 1.18 * omega  # the soil heat transfer coefficient
    sigma: float = 5.669e-8  # non-dimensional Stefan-Boltzmann constant

    # Geostrophic wind forcing
    u_G: float = 1.5  # u geostrophic wind
    v_G: float = 0.0  # v geostrophic wind

    latitude: float = 40  # latitude in grad
    f_c: float = 2 * 7.27 * 1e-5 * np.sin(latitude * np.pi / 180)  # coriolis parameter
    gamma: float = 0.01  # atmospheric lapse rate at upper edge of ABL in K/m

    EPS: float = 1E-16  # An imaginary numerical zero. Somehow the sqrt() of fenics needs this

    # closure specific parameters
    tau: float = 3600 * 5.0  # relaxation time scale
    min_tke: float = 10e-4  # minimum allowed TKE level
    Pr_t: float = 0.85  # turbulent Prandtl number
    alpha: float = 0.46  # eddy viscosity parametrization constant
    g: float = 9.81  # gravitational acceleration on Earth
    beta: float = g / T_ref  # for computing the Brunt-Vaisala frequency
    alpha_e: float = 0.1  # dissipation parametrization constant
    kappa: float = 0.41  # von Karman's constant


@dataclass
class Fenics_Parameters:
    pass


@dataclass
class Output_variables:
    pass


def initialize_project_variables():
    params = Parameters()
    fenics_params = Fenics_Parameters
    output = Output_variables

    return params, fenics_params, output
