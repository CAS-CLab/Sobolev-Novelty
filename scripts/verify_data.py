"""Check task lists, local SRBench tables and turbulence arrays."""

import argparse
from common import ROOT


def main():
    import numpy as np
    import pandas as pd
    from sn_dsrrans.data import DSRRANSData

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-external", action="store_true",
                   help="Require all SRBench tables in the task lists")
    a = p.parse_args()
    missing = []
    checked = 0
    for scope, count in [("whitebox", 133), ("blackbox", 122)]:
        tasks = (ROOT / f"data/srbench/{scope}.txt").read_text().split()
        if len(tasks) != count or len(set(tasks)) != count:
            raise ValueError(f"Invalid {scope} task list")
        for task in tasks:
            path = ROOT / f"data/pmlb/datasets/{task}/{task}.tsv.gz"
            if not path.is_file():
                missing.append(task)
                continue
            frame = pd.read_csv(path, sep="\t")
            if "target" not in frame or len(frame.columns) < 2:
                raise ValueError(f"Missing features or target: {path}")
            values = frame.dropna().to_numpy(dtype=float)
            if len(values) < 4 or not np.isfinite(values).all():
                raise ValueError(f"Invalid regression data: {path}")
            checked += 1
    if a.require_external and missing:
        raise FileNotFoundError(f"Missing {len(missing)} SRBench tables; first: {missing[0]}")
    turbulence = DSRRANSData.load(ROOT / "data/dsrrans")
    print(f"OK: task lists; {checked} SRBench tables; {len(turbulence.invariants)} turbulence rows.")
    print(f"SRBench tables not present: {len(missing)}")


if __name__ == "__main__":
    main()
