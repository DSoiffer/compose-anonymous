"""The eight families of 2D Gaussian base compositions (20 configurations each).

Every family has base P^0 = N(0, I) and varies one knob. The family keys follow
the names in the figures (DISPLAY_NAMES):

    anisotropic              P^A = N(0, diag(r, 1)), P^B = N(0, diag(1, r)), r in [1.05, 2.10]
    anisotropic_r_gt_2       the same, r log-spaced in [2, 200]
    orthogonal_shifts        P^A = N((m, 0), I), P^B = N((0, m), I), m in [0.3, 3]
    opposing_shifts          P^A = N((m, 0), I), P^B = N((-m, 0), I), m in [0.5, 6]
    shared_precision_p_lt_1  P^A = P^B = N(0, I/p), p from 0.95 to 0.73
    shared_precision_p_gt_1  the same, p log-spaced in [1.1, 16]
    pw_far_from_base         P^A = P^B = N((3, 3), s^2 I), s log-spaced from 0.6 to 0.03
    opposed_anisotropic      P^A = N(m u, diag(1.35, 1.05)), P^B = N(-m u, diag(1.05, 1.35)),
                             u = (1, -1)/sqrt(2), m in [1.2, 2.85]

Each configuration has a fixed seed (SEEDS) for its random draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensitivity_metric.gaussian.gaussian_tools import FullGaussian

# Family key -> display name used in the figures.
DISPLAY_NAMES = {
    "anisotropic": "Anisotropic sources",
    "anisotropic_r_gt_2": "Anisotropic sources, $r > 2$",
    "orthogonal_shifts": "Orthogonal shifts",
    "opposing_shifts": "Opposing shifts",
    "shared_precision_p_lt_1": "Shared precision, $p < 1$",
    "shared_precision_p_gt_1": "Shared precision, $p > 1$",
    "pw_far_from_base": r"$\mathbf{P}^w$ far from base",
    "opposed_anisotropic": "Opposed anisotropic sources",
}


# Seed of each configuration, in knob order. A configuration's TV samples use
# default_rng([seed, 0, 2]) and its k-th held-out draw default_rng([seed, 900 + k]).
SEEDS = {
    "anisotropic": [
        790480507, 1402437004, 1327268595, 1094945261, 634823681, 2674830694, 1125324063,
        2674223853, 1450798483, 1156422050, 4043212434, 3216863451, 1271644216, 2689804092,
        1943487431, 2977250406, 4008994359, 2383704278, 1120282280, 143721184
    ],
    "anisotropic_r_gt_2": [
        3793242197, 1327357482, 1343639219, 1969921811, 1162199574, 2879672329, 4275555904,
        1470804538, 1964511657, 2335007393, 3393566079, 814573040, 2259859093, 2801817795,
        2427994261, 165761686, 1004132506, 3855917954, 4031120464, 126719499
    ],
    "orthogonal_shifts": [
        3673549478, 4152738682, 3228756227, 4070926262, 2796213365, 1916160529, 3902632795,
        1712942691, 49853190, 3564642620, 1198850728, 3385964219, 3263338314, 488176660,
        1437696689, 2319945080, 283948082, 634832363, 2901937612, 299297075
    ],
    "opposing_shifts": [
        4033598063, 3259522885, 2258378762, 1380636241, 3089377142, 2256806670, 643093117,
        643765750, 1031986300, 3281074952, 1457707196, 732115043, 3916422337, 1264390139,
        3575739514, 876748869, 486847168, 1855902253, 2925345876, 2785256045
    ],
    "shared_precision_p_gt_1": [
        1365773462, 1587702678, 2752747005, 3679351247, 2877358096, 709913120, 445445445,
        362625165, 2195706615, 3574380918, 1888530516, 3519950093, 2676379828, 2406819322,
        3048240184, 3958940930, 1704219494, 1449816736, 2201593732, 2852799600
    ],
    "shared_precision_p_lt_1": [
        3991509182, 1543366004, 1074508975, 1572386387, 1113444558, 2252122724, 3177435323,
        4020250689, 1602074626, 647621885, 4124550680, 1105873808, 825193740, 2665937674,
        2747744757, 3299771666, 301965025, 165681709, 3702587870, 2584285189
    ],
    "pw_far_from_base": [
        1981022909, 1252102189, 3445872251, 3702481964, 1574770470, 2016121983, 3705173677,
        3859143429, 2869535498, 2614597208, 1487050829, 3517377441, 2171885677, 3448233283,
        1125495540, 143536727, 156713176, 51173091, 3490278962, 2385441326
    ],
    "opposed_anisotropic": [
        3410284744, 2764422366, 1345857142, 3827149073, 655113520, 256202100, 3104407505,
        1420991002, 3537177531, 1782753580, 1081754915, 910177201, 2036006625, 2742174538,
        2617950786, 3847701069, 434971608, 844721195, 2215986830, 4187972248
    ],
}


@dataclass
class Config:
    name: str
    family: str
    knob: float
    seed: int
    P0: FullGaussian
    PA: FullGaussian
    PB: FullGaussian

    def sources(self) -> dict[str, FullGaussian]:
        return {"0": self.P0, "A": self.PA, "B": self.PB}


def families() -> dict[str, list[Config]]:
    g = FullGaussian.diagonal
    u = np.array([1.0, -1.0]) / np.sqrt(2.0)
    # family -> (knob symbol, knob values, knob -> (P^A, P^B))
    specs = {
        "anisotropic": ("r", np.linspace(1.05, 2.10, 20),
                        lambda r: (g(0.0, [r, 1.0]), g(0.0, [1.0, r]))),
        "anisotropic_r_gt_2": ("r", np.geomspace(2, 200, 20),
                               lambda r: (g(0.0, [r, 1.0]), g(0.0, [1.0, r]))),
        "orthogonal_shifts": ("m", np.linspace(0.3, 3.0, 20),
                              lambda m: (g([m, 0.0], 1.0), g([0.0, m], 1.0))),
        "opposing_shifts": ("m", np.linspace(0.5, 6, 20),
                            lambda m: (g([m, 0.0], 1.0), g([-m, 0.0], 1.0))),
        "shared_precision_p_gt_1": ("p", np.geomspace(1.1, 16, 20),
                                    lambda p: (g(0.0, 1.0 / p), g(0.0, 1.0 / p))),
        "shared_precision_p_lt_1": ("p", np.linspace(0.95, 0.73, 20),
                                    lambda p: (g(0.0, 1.0 / p), g(0.0, 1.0 / p))),
        "pw_far_from_base": ("s", np.geomspace(0.6, 0.03, 20),
                             lambda s: (g(3.0, s * s), g(3.0, s * s))),
        "opposed_anisotropic": ("m", np.linspace(1.20, 2.85, 20),
                                lambda m: (g(m * u, [1.35, 1.05]), g(-m * u, [1.05, 1.35]))),
    }
    out = {}
    for family, (symbol, knobs, make) in specs.items():
        out[family] = [
            Config(f"{family}_{symbol}{float(k):g}", family, float(k), seed, g(0.0, 1.0),
                   *make(float(k)))
            for k, seed in zip(knobs, SEEDS[family])
        ]
    return out
