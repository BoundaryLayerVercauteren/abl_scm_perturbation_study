# standard imports
import matplotlib.pyplot as plt
from scipy import interpolate as scInt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py

# project related imports
from single_column_model.model import fenics_utility_functions as fut 
from single_column_model.model import surface_energy_balance as seb
from single_column_model.utils import save_solution as ss
from single_column_model.model import utility_functions as ut

def solution_loop(solver, params, output, fparams, u_n, v_n, T_n, k_n):

    Tg_n    = params.Tg_n
    
    T_D_low = fparams.theta_D_low
    #--------------------------------------------------------------------------
    print('Solving ... ')
    t = 0 # used for control
    
    i_w = 0 # index for writing
    for i in tqdm(range(params.SimEnd)):
        try:
            solver.solve()
        except:
            print("\n Solver crashed...")
            break
        
        # get variables to export
        us, vs, Ts, ks = fparams.uvTk.split(deepcopy=True)
        u_n.assign(us)
        v_n.assign(vs)
        T_n.assign(Ts)
        k_n.assign(ks)
        
        # control the minimum tke level to prevent sqrt(tke)
        k_n = set_minimum_tke_level(params, ks, k_n)
        

        phi1_fine_pr = project(fparams.f_ms, fparams.Q)
               
            
        #----------------------------------------------------------------------
        
        if (i)%params.Save_dt==0:
            # We first write out the variables, since the eddiy diffusivities gona be used anyway to update the boundary conditions.
            output = ss.save_current_result(output, params, fparams, i_w, us, vs, Ts, ks, phi1_fine_pr)
            i_w += 1

        
        # calc some variables
        u_now, v_now, Kh_now = ss.calc_variables_np(params, fparams, us, vs)
        
        # solve temperature at the ground
        Tg = seb.RHS_surf_balance_euler(Tg_n, fparams, params, Ts, u_now, v_now, Kh_now)
        
        # update temperature for the ODE
        Tg_n = np.copy(Tg)
        
        # update temperature for the PDE
        T_D_low.value = np.copy(Tg)
        
        # ugdate boundary conditions
        seb.update_tke_at_the_surface(fparams, params, u_now, v_now )
            
        i += 1
        t += params.dt


    return output
# =============================================================================


#============================ Common Functions================================
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    return np.abs(a - a0).argmin()


def set_minimum_tke_level(params, ks, k_n):
    
    # "ks" is the current solution
    # "k_n" the initial value for the next iteration
    
    # limiting the value by converting to numpy. Hmm.. there is must be a better way.
    ks_array = ks.vector().get_local()
    
    # set back a to low value
    ks_array[ks_array < params.min_tke] = params.min_tke
    
    # cast numpy to fenics variable
    ks.vector().set_local(ks_array)
    
    #update the value for the next simulation run
    k_n.assign(ks)
    
    return k_n
    









