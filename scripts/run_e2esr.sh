#!/usr/bin/env bash
set -euo pipefail

E2ESR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${E2ESR_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" -B -m src.generative.e2esr.train \
  --pretrain_filter sobolev_novelty \
  --sn_threshold 0.31622776601683794 \
  --sn_geometry_rows 200 \
  --sn_min_valid_rows 32 \
  --sn_value_weight 1.0 \
  --sn_gradient_weight 1.0 \
  --sn_geometry_seed 20260806 \
  "$@"
