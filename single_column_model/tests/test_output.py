import os
import shutil
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import pytest
from dataclasses_json import dataclass_json

from single_column_model.model import run_model


def pytest_configure():
    pytest.path_output = None
    pytest.data_file_name = None
    pytest.theta = None
    pytest.u = None
    pytest.v = None
    pytest.tke = None
    pytest.z = None


def test_successful_run():
    try:
        pytest.path_output = run_model.run_model()
    except Exception as exc:
        assert False, f"Model run raised an exception: {exc}"


def test_data_file_exists():
    for file in os.listdir(pytest.path_output):
        if "solution_" in file:
            pytest.data_file_name = file

    if not pytest.data_file_name:
        assert False, "No output file produced from previous test."


def test_data_has_no_nan():
    with h5py.File(pytest.path_output + pytest.data_file_name, "r+") as file:
        t = file["t"][:]
        z = file["z"][:]
        u = file["u"][:]
        v = file["v"][:]
        theta = file["theta"][:]
        tke = file["TKE"][:]

    if (
        np.isnan(t).any()
        or np.isnan(z).any()
        or np.isnan(u).any()
        or np.isnan(v).any()
        or np.isnan(theta).any()
        or np.isnan(tke).any()
    ):
        assert False, "At least one output variable contains a NaN."

    pytest.theta = pd.DataFrame(data=theta, columns=t.flatten(), index=z.flatten())
    pytest.u = pd.DataFrame(data=u, columns=t.flatten(), index=z.flatten())
    pytest.v = pd.DataFrame(data=v, columns=t.flatten(), index=z.flatten())
    pytest.tke = pd.DataFrame(data=tke, columns=t.flatten(), index=z.flatten())

    pytest.z = z.flatten()

    assert True


@pytest.fixture(scope="session")
def load_parameter_from_file():
    @dataclass_json
    @dataclass
    class Parameters:
        theta_ref: float
        gamma: float
        u_G: float
        kappa: float
        z0: float

    for file in os.listdir(pytest.path_output):
        if "parameters" in file:
            with open(pytest.path_output + file, "r") as file:
                param_data = file.read()

    return Parameters.from_json(param_data)


def test_initial_cond_u():
    params = load_parameter_from_file()
    c_f = 4 * 10 ** (-3)
    u_star_ini = np.sqrt(0.5 * c_f * params.u_G**2)
    u_0 = u_star_ini / params.kappa * np.log(pytest.z / params.z0)
    assert np.all(np.isclose(pytest.u.iloc[:, 0].values, u_0, atol=1e-2))


def test_initial_cond_v():
    assert np.all(np.isclose(pytest.v.iloc[:, 0].values, 0.0, atol=1e-3))


def test_initial_cond_theta():
    params = load_parameter_from_file()
    z_cut = 200
    z_idx = (np.abs(pytest.z - z_cut)).argmin()
    theta_above_z_cut = (
        params.theta_ref + (pytest.z[z_idx + 1 :] - z_cut) * params.gamma
    )
    assert np.all(
        np.isclose(pytest.theta.iloc[:z_idx, 0].values, params.theta_ref, rtol=1e-3)
    )
    assert np.all(
        np.isclose(
            pytest.theta.iloc[z_idx + 1 :, 0].values, theta_above_z_cut, rtol=1e-3
        )
    )


def test_initial_cond_tke():
    tke_H_t0 = 0

    params = load_parameter_from_file()
    c_f = 4 * 10 ** (-3)
    u_star_ini = np.sqrt(0.5 * c_f * params.u_G**2)
    tke_z0_t0 = u_star_ini**2 / np.sqrt(0.087)

    a = (tke_H_t0 - tke_z0_t0) / (np.log(pytest.z[-1]) - np.log(pytest.z[0]))
    b = tke_z0_t0 - a * np.log(pytest.z[0])

    tke_init = a * np.log(pytest.z + pytest.z[0]) + b

    assert np.all(np.isclose(pytest.tke.iloc[:, 0].values, tke_init, atol=1e-1))


def test_lower_boundary_cond_u():
    assert np.all(np.isclose(pytest.u.iloc[0, :].values, 0.0, atol=1e-3))


def test_lower_boundary_cond_v():
    assert np.all(np.isclose(pytest.v.iloc[0, :].values, 0.0, atol=1e-3))


# def test_lower_boundary_cond_theta():
#     params = load_parameter_from_file()
#     assert np.all(np.isclose(pytest.theta.iloc[0,:].values, params.theta_ref, rtol=1e-2))


def test_lower_boundary_cond_tke():
    params = load_parameter_from_file()
    c_f = 4 * 10 ** (-3)
    u_star_ini = np.sqrt(0.5 * c_f * params.u_G**2)
    tke_low = u_star_ini**2 / np.sqrt(0.087)
    assert np.all(np.isclose(pytest.tke.iloc[0, :].values, tke_low, atol=1e-1))


def test_upper_boundary_cond_u():
    params = load_parameter_from_file()
    gradient = (pytest.u.iloc[-1, :] - pytest.u.iloc[-2, :]) / (
        pytest.z[-1] - pytest.z[-2]
    )
    assert np.all(np.isclose(pytest.u.iloc[-1, :].values, params.u_G, atol=1e-1))
    assert np.all(np.isclose(gradient.values, 0.0, atol=1e-3))


def test_upper_boundary_cond_v():
    assert np.all(np.isclose(pytest.v.iloc[-1, :].values, 0.0, atol=1e-3))


def test_upper_boundary_cond_theta():
    params = load_parameter_from_file()
    gradient = (pytest.theta.iloc[-1, :] - pytest.theta.iloc[-2, :]) / (
        pytest.z[-1] - pytest.z[-2]
    )
    assert np.all(np.isclose(gradient.values, params.gamma, atol=1e-3))


def test_upper_boundary_cond_tke():
    gradient = (pytest.tke.iloc[-1, :] - pytest.tke.iloc[-2, :]) / (
        pytest.z[-1] - pytest.z[-2]
    )
    assert np.all(np.isclose(gradient.values, 0.0, atol=1e-3))


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup the testing directory once all tests were run."""

    def remove_test_dir():
        shutil.rmtree(pytest.path_output)

    request.addfinalizer(remove_test_dir)
