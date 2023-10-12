def set_minimum_tke_level(params, ks, k_n):
    """Set minimal value for TKE to prevent crashing."""
    # Note: "ks" is the current solution and "k_n" the initial value for the next iteration

    # Transform fenics variable to array
    ks_array = ks.vector().get_local()

    # Set all values which are below minimal at allowed TKE to this pre-defined minimum
    ks_array[ks_array < params.min_tke] = params.min_tke

    # Transform array back to fenics variable
    ks.vector().set_local(ks_array)

    # Update the TKE value for the next simulation run
    k_n.assign(ks)

    return k_n
