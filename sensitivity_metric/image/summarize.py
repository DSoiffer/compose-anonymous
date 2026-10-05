"""Turn estimate_ratios.py outputs into -log rESS tables.

For each source i the composition log weights are

    log w_0 = d_A0 + d_B0   on samples from P^0
    log w_A = d_B0          on samples from P^A
    log w_B = d_A0          on samples from P^B

and -log rESS_i = log(sum w^2) + log N - 2 log(sum w), computed from the raw
estimates without any Monte Carlo correction.

Pass one file per cell, labeled CELL=PATH, with cells named <pair>_<ID|OOD>:

    python -m sensitivity_metric.image.summarize FC_ID=ratios/room_FC_ID_trimmed.npz FC_OOD=... NFC_ID=... NFC_OOD=...
"""

from __future__ import annotations

import argparse

import numpy as np

from sensitivity_metric.ress import neg_log_ress


def cell_values(path: str) -> dict[str, float]:
    z = np.load(path)
    log_w = {
        "0": z["d_A0_at_0"] + z["d_B0_at_0"],
        "A": z["d_B0_at_A"],
        "B": z["d_A0_at_B"],
    }
    values = {s: neg_log_ress(v) for s, v in log_w.items()}
    values["max_AB"] = max(values["A"], values["B"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="+", metavar="CELL=PATH")
    args = parser.parse_args()

    rows = {}
    for item in args.cells:
        cell, path = item.split("=", 1)
        rows[cell] = cell_values(path)

    print("| Setting | -log rESS_0 | -log rESS_a1 | -log rESS_a2 | max over a1, a2 |")
    print("|---|---:|---:|---:|---:|")
    for cell, v in rows.items():
        print(f"| {cell} | {v['0']:.2f} | {v['A']:.2f} | {v['B']:.2f} | {v['max_AB']:.2f} |")

    print("\nID/OOD gap in max over a1, a2 (OOD minus ID):")
    for pair in sorted({cell.rsplit('_', 1)[0] for cell in rows}):
        if f"{pair}_ID" in rows and f"{pair}_OOD" in rows:
            gap = rows[f"{pair}_OOD"]["max_AB"] - rows[f"{pair}_ID"]["max_AB"]
            print(f"  {pair}: {gap:+.2f}")


if __name__ == "__main__":
    main()
