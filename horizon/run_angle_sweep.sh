#!/bin/bash
# ===========================================
# Sweep over horizon angle ranges for RDueling experiments
# Usage:
#   bash run_angle_sweep.sh
# ===========================================

for min in 5 10 15 20 25 30 35 40; do
  max=$((min+5))
  echo ">>> Running for range [$min, $max] ..."
  python main.py --min_deg $min --max_deg $max --save --show false
done

echo "All experiments completed"
