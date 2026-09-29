import importlib.util

import numpy as np
import pandas as pd

from battery_rul.data import discharge_segments, discover, load_calce_cell, load_nasa_mat
from conftest import ROOT


def _generator():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "make_synthetic_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arbin(caps, cumulative=True, cycle_index_constant=False):
    rows, cum = [], 0.0
    for k, c in enumerate(caps, start=1):
        cyc = 1 if cycle_index_constant else k
        rows.append({"Cycle_Index": cyc, "Current(A)": 0.5, "Voltage(V)": 4.1, "Step_Time(s)": 10,
                     "Discharge_Capacity(Ah)": cum if cumulative else 0.0})
        run = 0.0
        for j in range(4):
            if cumulative:
                cum += c / 4
                value = cum
            else:
                run += c / 4
                value = run
            rows.append({"Cycle_Index": cyc, "Current(A)": -0.5, "Voltage(V)": 4.0 - 0.2 * j,
                         "Step_Time(s)": 10 * j, "Discharge_Capacity(Ah)": value})
    return pd.DataFrame(rows)


def test_segments_recover_capacity_from_cumulative_column():
    caps = np.array([1.30, 1.29, 1.27, 1.26])
    got = discharge_segments(_arbin(caps, cumulative=True))["capacity_Ah"].to_numpy()
    np.testing.assert_allclose(got, caps)


def test_segments_with_constant_cycle_index():
    caps = np.array([1.1, 1.05, 1.0])
    got = discharge_segments(_arbin(caps, cumulative=True, cycle_index_constant=True))["capacity_Ah"]
    np.testing.assert_allclose(got.to_numpy(), caps)


def test_segments_with_resetting_column():
    caps = np.array([1.2, 1.1])
    got = discharge_segments(_arbin(caps, cumulative=False))["capacity_Ah"].to_numpy()
    np.testing.assert_allclose(got, caps)


def test_synthetic_files_round_trip(tmp_path):
    gen = _generator()
    rng = np.random.default_rng(0)
    gen.write_nasa(tmp_path, rng)
    gen.write_calce(tmp_path, rng, n_cycles=60)
    found = discover(tmp_path)
    assert {"B0005", "B0018", "CX2_34"} <= set(found)
    nasa = load_nasa_mat(found["B0005"][1])
    assert len(nasa) == 168 and nasa["capacity_Ah"].between(1.0, 2.2).all()
    calce = load_calce_cell(found["CX2_34"][1])
    assert calce["capacity_Ah"].between(0.4, 1.5).all()      # never running totals
    assert not calce["source_file"].str.contains("self discharge").any()
    assert (calce["cycle"] == np.arange(1, len(calce) + 1)).all()
