"""The 64-permutation momentum matrix: the six blueprint upgrades (A–F) as a
config, run walk-forward over the survivorship-corrected universe, split into
train/validation, with a small-account feasibility calc. Pure (data in, numbers
out); the build script supplies prices/sectors/benchmark/pit.
"""
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class MomentumConfig:
    vol_adjust: bool = False        # A
    sector_neutral: bool = False    # B
    trend_filter: bool = False      # C
    slots: int = 15                 # D: 15 off / 10 on
    freq: str = "M"                 # E: "M" off / "Q" on
    lazy: bool = False              # F

    @property
    def code(self) -> str:
        flags = [self.vol_adjust, self.sector_neutral, self.trend_filter,
                 self.slots == 10, self.freq == "Q", self.lazy]
        return "".join(c if on else "·" for c, on in zip("ABCDEF", flags))

    def kwargs(self) -> dict:
        return dict(vol_adjust=self.vol_adjust, sector_neutral=self.sector_neutral,
                    trend_filter=self.trend_filter, k=self.slots, freq=self.freq,
                    lazy=self.lazy)


ALL_CONFIGS = [MomentumConfig(va, sn, tf, slots, freq, lz)
               for va, sn, tf, slots, freq, lz in product(
                   (False, True), (False, True), (False, True),
                   (15, 10), ("M", "Q"), (False, True))]
