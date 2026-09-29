"""Final Tensor-SN-DSRRANS forward selection and optional Base-loss pruning."""

import argparse
import subprocess
import sys
from common import ROOT, new_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["base", "sn"], required=True)
    p.add_argument("--seed", type=int, default=1000003)
    p.add_argument("--fit-fraction", type=float, choices=[0.75, 1.0], default=0.75)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--prune-to",
        type=int,
        choices=[12, 20, 30, 40, 60, 80],
        help="Base-loss backward pruning of the selected 100-term support",
    )
    a = p.parse_args()
    out = new_output(a.output)
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/dsrrans_forward.py"),
        "--dataset-dir",
        str(ROOT / "data/dsrrans"),
        "--output-dir",
        str(out / "forward"),
        "--max-degree",
        "12",
        "--max-terms",
        "3" if a.smoke else "100",
        "--methods",
        "base" if a.method == "base" else "sn_epsilon",
        "--geometry-samples",
        "9600",
        "--geometry-mode",
        "tensor_sobolev",
        "--geometry-gradient-weight",
        "0.005",
        "--epsilon",
        "0.06",
        "--fit-fraction",
        str(a.fit_fraction),
        "--split-seed",
        str(a.seed),
        "--skip-degree-curve",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    if a.prune_to:
        if a.smoke:
            raise ValueError("Pruning requires the full 100-term run")
        method = "base" if a.method == "base" else "sn_epsilon"
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts/dsrrans_backward.py"),
                "--dataset-dir",
                str(ROOT / "data/dsrrans"),
                "--output-dir",
                str(out / "pruned"),
                "--initial-trace",
                str(out / f"forward/{method}_trace.csv"),
                "--initial-terms",
                "100",
                "--max-degree",
                "12",
                "--min-terms",
                str(a.prune_to),
                "--methods",
                "base",
                "--snapshot-terms",
                str(a.prune_to),
                "--fit-fraction",
                str(a.fit_fraction),
                "--split-seed",
                str(a.seed),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
