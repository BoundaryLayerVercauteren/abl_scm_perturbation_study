# standard imports
from tqdm import tqdm
import fenics as fe
import numpy as np
import traceback

# project related imports
from single_column_model.model import surface_energy_balance as seb
from single_column_model.model import define_PDE_model as dPm
from single_column_model.utils import save_solution as ss
from single_column_model.utils import transform_values as tv



def solution_loop(solver, params, output, fenics_params, u_n, v_n, T_n, k_n):
    Tg_n = params.Tg_n
    T_D_low = fenics_params.theta_D_low
    # --------------------------------------------------------------------------
    print('Solving ... ')
    t = 0  # used for control
    i_w = 0  # index for writing

    for i in tqdm(range(params.num_steps)):

        # Add perturbation value for current time step to PDE
        if 'pde' in params.perturbation_param:
            cur_perturbation = tv.convert_numpy_array_to_fenics_function(output.perturbation[:, i], fenics_params.Q)
            if params.perturbation_param == 'pde_u':
                perturbed_F = fenics_params.F - cur_perturbation * fenics_params.u_test * fe.dx
            elif params.perturbation_param == 'pde_theta':
                perturbed_F = fenics_params.F - cur_perturbation * fenics_params.theta_test * fe.dx
            solver = dPm.prepare_fenics_solver(fenics_params, perturbed_F)

        try:
            solver.solve()
        except:
            print("\n Solver crashed...")
            print(traceback.format_exc())
            break

        # get variables to export
        us, vs, Ts, ks = fenics_params.uvTk.split(deepcopy=True)
        u_n.assign(us)
        v_n.assign(vs)
        T_n.assign(Ts)
        k_n.assign(ks)

        # control the minimum tke level to prevent sqrt(tke)
        k_n = set_minimum_tke_level(params, ks, k_n)

        phi1_fine_pr = fe.project(fenics_params.f_ms, fenics_params.Q)

        # ----------------------------------------------------------------------

        if (i) % params.save_dt_sim == 0:
            # We first write out the variables, since the eddiy diffusivities gona be used anyway to update the boundary conditions.
            output = ss.save_current_result(output, params, fenics_params, i_w, us, vs, Ts, ks, phi1_fine_pr)
            i_w += 1

        # calc some variables
        u_now, v_now, Kh_now = ss.calc_variables_np(params, fenics_params, us, vs)

        # solve temperature at the ground
        Tg = seb.RHS_surf_balance_euler(Tg_n, fenics_params, params, Ts, u_now, v_now, Kh_now)

        # update temperature for the ODE
        Tg_n = np.copy(Tg)

        # update temperature for the PDE
        T_D_low.value = np.copy(Tg)

        # ugdate boundary conditions
        seb.update_tke_at_the_surface(fenics_params, params, u_now, v_now)

        i += 1
        t += params.dt

    return output


# =============================================================================


# ============================ Common Functions================================
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

    # update the value for the next simulation run
    k_n.assign(ks)

    return k_n
