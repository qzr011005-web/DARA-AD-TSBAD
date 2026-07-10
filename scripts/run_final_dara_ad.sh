#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):${PYTHONPATH}"

source configs/final_dara_ad.env

rm -f eval/metrics/multi/DeCo.csv
rm -rf eval/score/multi/DeCo

python benchmark_exp/Run_DeCo_Detector_M.py
