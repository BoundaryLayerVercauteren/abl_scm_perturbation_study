# standard imports
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
import os


# -----------------------------------------------------------------------------
@dataclass_json
@dataclass
class Parameters:
    save_ini_cond: bool = True  # save simulations solution as initial condition
    load_ini_cond: bool = False  # load existing initial condition

    T_end_h: float = 2  # hour
    T_end: float = T_end_h * 3600  # seconds
    dt: float = 1  # seconds; make good choice: (1, 2, 3, 4, 5, 6, 10, 20) / (10)
    SimEnd: int = int(T_end / dt)  # number of time steps

    # file name for initial conditions
    init_cond_path: str = 'T_end_' + str(T_end_h) + 'h_'

    # what time steps to save
    SimSav: float = 1  # in seconds
    Save_dt: int = int(SimSav / dt)
    Save_tot: int = int(T_end / SimSav)

    Nz: int = 100  # number of point/ domain resolution
    s_Nz: int = 1024  # number of points in stochastic domain
    z0: float = 0.044  # roughness length in meter
    z0h: float = z0 * 0.1  # roughness length for heat in meter
    H: float = 300.0  # domain height in meters  ! should be H > z_l * s_dom_ext

    omega: float = (2 * np.pi) / (24 * 60 * 60)  # angular earth velocity
    theta_m: float = 290  # restoring temperature of peat soil
    T_ref: float = 300  # reference potetial temperature [K]
    rho: float = 1.225  # air density kg/m**3 at 15 C
    C_p: float = 1005  # specific heat capacity at constant pressure of air
    C_g: float = 0.95 * (1.45 * 3.58 * 1e+6 / 2 / omega) ** 0.5  # heat capacity of ground per unit area
    sig: float = 5.669e-8  # non-dimensional Stefan-Boltzmann constant
    Qc: float = 0.0  # the cloud fraction
    Qa: float = 0.003  # specific humidity [g kg^-1]
    Tg_n: float = 300  # temperature initial value at the ground [K]. Will be set later
    R_n: float = -30
    k_m: float = 1.18 * omega  # the soil heat transfer coefficient

    # Geostrophic wind forcing. If V_g not "0.0" one need new initial conditions
    u_G: float = 1.5  # u geostrophic wind
    v_G: float = 0.0  # v geostrophic wind

    latitude: float = 40  # latitude in grad
    f_c: float = 2 * 7.27 * 1e-5 * np.sin(latitude * np.pi / 180)  # coriolis parameter
    gamma: float = 0.01  # atmospheric lapse rate at upper edge of ABL in K/m

    EPS: float = 1E-16  # An imaginary numerical zero. Somehow the sqrt() of fenics needs this

    # closure specific parameters
    tau: float = 3600 * 1  # relaxation time scale
    min_tke: float = 1e-4  # minimum allowed TKE level
    Pr_t: float = 0.85  # turbulent Prandtl number
    alpha: float = 0.46  # eddy viscosity parametrization constant
    g: float = 9.81  # gravitational acceleration on Earth
    beta: float = g / T_ref  # for computing the Brunt-Vaisala frequency
    alpha_e: float = 0.1  # dissipation parametrization constant
    kappa: float = 0.41  # von Karman's constant

    # stochastic model specific parameter 
    d: float = -0.07  # submesoscale intensity. d=0 is estimated from FLOSS2 data set
    z_l: float = 50  # height [m] till the stochastic model is active. Above the classical mixing is active
    lz: float = 20  # covariance length in height [m]
    s_dom_ext: float = 2.0  # this is by how much the height of the stochastic domaint is extended.  Needs to include bleding height.

    # methods
    def update(self):
        self.set_T_end(self)
        self.set_SimEnd(self)
        self.set_z0h(self)
        self.set_omega(self)
        self.set_k_m(self)
        self.set_C_g(self)
        self.set_f_c(self)
        self.set_beta(self)

    def set_beta(self):
        self.beta = self.g / self.T_ref

    def set_f_c(self):
        self.f_c = 2 * 7.27 * 1e-5 * np.sin(self.latitude * np.pi / 180)

    def set_C_g(self):
        self.C_g = 0.95 * (1.45 * 3.58 * 1e+6 / 2 / self.omega) ** 0.5

    def set_k_m(self):
        self.k_m = 1.18 * self.omega

    def set_omega(self):
        self.omega = (2 * np.pi) / (24 * 60 * 60)

    def set_z0h(self):
        self.z0h = self.z0 * 0.1

    def set_SimEnd(self):
        self.SimEnd = int(self.T_end / self.dt)

    def set_T_end(self):
        self.T_end = self.T_end_h * 3600

    # remove last save if present
    # try:
    #     os.remove('solution/' + solStringName + '.h5')
    # except OSError:
    #     pass


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
@dataclass
class Fenics_Parameters:
    pass


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
@dataclass
class Output_variables:
    pass


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def initialize_project_variables():
    params = Parameters()
    fparams = Fenics_Parameters
    output = Output_variables

    return params, fparams, output
# -----------------------------------------------------------------------------
