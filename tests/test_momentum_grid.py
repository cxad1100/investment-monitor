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


import numpy as np
import pandas as pd
from tools.momentum_grid import run_grid


def _grid_px(n=900, ncols=12, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = {f"T{i}": 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
            for i in range(ncols)}
    return pd.DataFrame(cols, idx)


def test_run_grid_splits_train_val_and_covers_configs():
    px = _grid_px()
    slip = {t: 10 for t in px.columns}
    configs = [MomentumConfig(), MomentumConfig(slots=10)]      # 2 of the 64
    res = run_grid(px, slip, sectors={t: "X" for t in px.columns},
                   configs=configs, train_end="2020-06-30",
                   lookback=200, skip=10)
    assert {c["code"] for c in res["cells"]} == {"······", "···D··"}
    cell = res["cells"][0]
    for part in ("train", "val", "full"):
        assert "sharpe" in cell[part] and "net_return" in cell[part]
    assert cell["trades_per_year"] >= 0
