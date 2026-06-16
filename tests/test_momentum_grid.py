from tools.momentum_grid import MomentumConfig, ALL_CONFIGS


def test_all_configs_is_64_unique():
    assert len(ALL_CONFIGS) == 64
    assert len({c.code for c in ALL_CONFIGS}) == 64


def test_baseline_config_code_and_kwargs():
    base = MomentumConfig()
    assert base.code == "······"                          # all upgrades off
    kw = base.kwargs()
    assert kw["k"] == 15 and kw["freq"] == "M"
    assert kw["vol_adjust"] is False and kw["lazy"] is False
    full = MomentumConfig(True, True, True, 10, "Q", True)
    assert full.code == "ABCDEF"
    assert full.kwargs()["k"] == 10 and full.kwargs()["freq"] == "Q"
