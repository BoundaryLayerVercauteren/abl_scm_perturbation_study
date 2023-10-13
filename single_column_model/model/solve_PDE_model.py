# coding=utf-8
# !/usr/bin/env python

import traceback

import fenics as fe
import numpy as np
from scipy import interpolate

from single_column_model.model import (control_tke,
                                       define_initial_and_boundary_conditions,
                                       define_PDE_model,
                                       define_stochastic_part,
                                       surface_energy_balance)
from single_column_model.utils import save_solution, transform_values


def add_perturbation_to_weak_form_of_model(
        perturbation_param, perturbation, Q, det_weak_form, u_test, theta_test, idx
):
    """Function to add perturbation values for the current time step to the weak formulation of the PDE model."""
    if "pde" in perturbation_param:
        # Transform perturbation values for current time step to fenics function. This is necessary when the PDEs
        # themselves are perturbed
        cur_perturbation = transform_values.convert_numpy_array_to_fenics_function(
            perturbation[:, idx], Q
        )
        # Add perturbation to one of the differential equations in the weak form.
        if perturbation_param == "pde_u":
            perturbed_weak_form = det_weak_form - cur_perturbation * u_test * fe.dx
        elif perturbation_param == "pde_theta":
            perturbed_weak_form = det_weak_form - cur_perturbation * theta_test * fe.dx
        else:
            raise SystemExit(f"\n The given perturbation ({perturbation_param}) is not defined.")

    # The differential equations are not perturbed in this case. This holds e.g. for a perturbed net radiation.
    elif perturbation_param == "net_rad":
        cur_perturbation = perturbation[:, idx]
        perturbed_weak_form = det_weak_form

    # No perturbation is added in this case
    else:
        cur_perturbation = 0.0
        perturbed_weak_form = det_weak_form

    return cur_perturbation, perturbed_weak_form


def solve_stoch_stab_func_current_time_step(fenics_param, model_param):
    # Calculate richardson number for current time step
    richardson_num_det = define_PDE_model.Ri(fenics_param, model_param)
    # Interpolate richardson number from deterministic to stochastic grid
    interpolation_func = interpolate.interp1d(fenics_param.z[:, 0], richardson_num_det, fill_value='extrapolate')
    richardson_num_stoch = interpolation_func(model_param.stoch_grid)[:, 0]
    # Get parameter for stability function
    Lambda, Upsilon, Sigma = define_stochastic_part.get_stoch_stab_function_parameter(richardson_num_stoch)


def solution_loop(params, output, fenics_params, u_n, v_n, theta_n, k_n):
    theta_g_n = params.theta_g_n

    # --------------------------------------------------------------------------
    print("Solving PDE system ... ")
    saving_idx = 0  # index for writing

    for time_idx in range(params.num_steps):

        if params.perturbation_param != 'stab_func':
            # Add perturbation to weak formulation of PDE model
            perturbation_at_time_idx, perturbed_F = add_perturbation_to_weak_form_of_model(
                params.perturbation_param,
                output.perturbation,
                fenics_params.Q,
                fenics_params.F,
                fenics_params.u_test,
                fenics_params.theta_test,
                time_idx,
            )
            # Solve PDE model with Finite Element Method in space
            solver = define_PDE_model.prepare_fenics_solver(fenics_params, perturbed_F)

        try:
            solver.solve()
        except Exception:
            print("\n Solver crashed due to ...")
            print(traceback.format_exc())
            break

        # Get variables to export
        u_sol, v_sol, theta_sol, k_sol = fenics_params.uvTk.split(deepcopy=True)
        u_n.assign(u_sol)
        v_n.assign(v_sol)
        theta_n.assign(theta_sol)
        k_n.assign(k_sol)

        # Control the minimum TKE level to prevent sqrt(TKE) crashing
        k_n = control_tke.set_minimum_tke_level(params, k_sol, k_n)

        if params.perturbation_param == 'stab_func':
            pass  # fenics_params.f_ms.value = solve_stoch_stab_func_current_time_step()

        # Save solution at current time step if it fits with the saving time interval
        if time_idx % params.save_dt_sim == 0:
            output = save_solution.save_current_result(output, params, fenics_params, saving_idx, u_sol, v_sol,
                                                       theta_sol, k_sol)
            saving_idx += 1

        # Transform values to numpy arrays
        u_sol_np = transform_values.interpolate_fenics_function_to_numpy_array(u_sol, fenics_params.Q)
        v_sol_np = transform_values.interpolate_fenics_function_to_numpy_array(v_sol, fenics_params.Q)

        # Get corresponding heat flux
        Kh_sol_np = transform_values.project_fenics_function_to_numpy_array(define_PDE_model.K_h(fenics_params, params),
                                                                            fenics_params.Q)

        # Calculate surface temperature for current time step
        theta_g = surface_energy_balance.calculate_surface_temperature_euler_form(
            theta_g_n,
            fenics_params,
            params,
            theta_sol,
            Kh_sol_np,
            perturbation_at_time_idx,
        )

        # Update corresponding variable (ODE)
        theta_g_n = np.copy(theta_g)

        # Update lower boundary condition for potential temperature
        fenics_params.theta_D_low.value = np.copy(theta_g)

        # Update lower boundary condition for TKE
        fenics_params.k_D_low.value = define_initial_and_boundary_conditions.update_tke_at_the_surface(params.kappa,
                                                                                                       fenics_params.z,
                                                                                                       u_sol_np,
                                                                                                       v_sol_np,
                                                                                                       params.min_tke)

        time_idx += 1

    return output
