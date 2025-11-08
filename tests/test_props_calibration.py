import math

import pytest

pytest.importorskip("numpy")

from cfb.player_prop_sim import (
    RoleParams,
    YardageModel,
    simulate_receiving,
    simulate_rushing_yards,
)
from cfb.props_calibration import load_calibrations


@pytest.fixture(scope="module")
def calibrations():
    return load_calibrations()


def test_calibration_bucket_resolution(calibrations):
    wr_entry = calibrations.get_volume("receiving", ["WR1", "DEFAULT"], mode="auto")
    assert wr_entry is not None
    assert math.isclose(wr_entry.phi, 0.0)

    # rb1 bucket has only one sample → excluded in auto mode
    rb_auto = calibrations.get_volume("receiving", ["RB1"], mode="auto")
    assert rb_auto is None

    rb_forced = calibrations.get_volume("receiving", ["RB1"], mode="calibrated")
    assert rb_forced is not None


def test_receiving_calibration_overrides_phi(calibrations):
    role = RoleParams(
        target_share_mean=0.22,
        target_share_kappa=45.0,
        position="TE",
        role="TE1",
    )
    yard_model = YardageModel(mean=12.0, sd=4.0)

    heur = simulate_receiving(
        tgt_mean=6.0,
        catch_rate=0.64,
        yards_per_rec=yard_model,
        n_sims=4000,
        seed=123,
        role=role,
        calibration_mode="heuristic",
    )

    calibrated = simulate_receiving(
        tgt_mean=6.0,
        catch_rate=0.64,
        yards_per_rec=yard_model,
        n_sims=4000,
        seed=123,
        role=role,
        calibrations=calibrations,
        calibration_mode="calibrated",
    )

    cal_entry = calibrations.get_volume("receiving", ["TE1", "TE", "DEFAULT"], mode="calibrated")
    assert cal_entry is not None

    heur_phi = heur["yards"]["overdispersion_phi"]
    cal_phi = calibrated["yards"]["overdispersion_phi"]
    assert cal_phi == pytest.approx(cal_entry.phi)
    assert not math.isclose(heur_phi, cal_phi)

    assert calibrated["yards"]["zero_inflation"] == pytest.approx(cal_entry.zero_inflation)


def test_rushing_calibration_adjusts_variance(calibrations):
    role = RoleParams(position="WR")
    yard_model = YardageModel(mean=5.0, sd=2.0)

    heur = simulate_rushing_yards(
        rush_mean=8.0,
        yards_per_rush=yard_model,
        n_sims=4000,
        seed=321,
        role=role,
        is_qb=False,
        calibration_mode="heuristic",
    )

    calibrated = simulate_rushing_yards(
        rush_mean=8.0,
        yards_per_rush=yard_model,
        n_sims=4000,
        seed=321,
        role=role,
        is_qb=False,
        calibrations=calibrations,
        calibration_mode="calibrated",
    )

    cal_entry = calibrations.get_volume("rushing", ["WR", "DEFAULT"], mode="calibrated")
    assert cal_entry is not None

    heur_phi = heur["yards"]["overdispersion_phi"]
    cal_phi = calibrated["yards"]["overdispersion_phi"]
    assert cal_phi == pytest.approx(cal_entry.phi)
    assert not math.isclose(heur_phi, cal_phi)
