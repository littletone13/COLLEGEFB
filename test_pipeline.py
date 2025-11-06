
# tests/test_pipeline.py
import importlib
import math

def test_names_normalize():
    names = importlib.import_module("cfb.names")
    assert names.normalize_player("Marv\u00edn Harrison Jr.") == "MARVIN HARRISON JR."
    assert names.normalize_team("lSu") == "LSU"
    assert names.normalize_team("  Ohio  State ") == "OHIO STATE"

def test_simulate_smoke():
    sim = importlib.import_module("cfb_player_sim") if importlib.util.find_spec("cfb_player_sim") else importlib.import_module("cfb.player_prop_sim")
    PassingParams = getattr(sim, "PassingParams")
    ReceivingParams = getattr(sim, "ReceivingParams")
    RushingParams = getattr(sim, "RushingParams")
    Lines = getattr(sim, "Lines")
    res = sim.simulate_player(
        passing=PassingParams(att_mean=30, comp_rate=0.65, yds_per_comp_mu=11.0, yds_per_comp_sd=5.5),
        receiving=ReceivingParams(tgt_mean=8.0, catch_rate=0.67, yds_per_rec_mu=12.5, yds_per_rec_sd=6.5),
        rushing=RushingParams(rush_mean=15.0, yds_per_rush_mu=4.8, yds_per_rush_sd=2.2),
        lines=Lines(pass_yds=250.5, receptions=6.5, rush_yds=70.5),
        sims=5000, seed=123
    )
    # sanity: keys exist and probabilities are 0..1
    for k in ["pass_yds","receptions","rush_yds"]:
        assert k in res and 0 <= res[k].get("Pr(> line)", 0.5) <= 1

def test_report_writer_imports():
    # ensure write_report and pricing import (for Makefile targets)
    import write_report
    import pricing
