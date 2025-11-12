#!/bin/bash
# ===========================================
# Sweep over horizon angle ranges for RDueling experiments
# Usage:
#   bash run_angle_sweep.sh
# ===========================================

#!/bin/bash

for min in $(seq 5 2.5 40); do
  max=$(echo "$min + 2.5" | bc)
  echo ">>> Running for range [$min, $max] ..."
  python main.py --min_deg $min --max_deg $max --save --show false
done

echo "All experiments completed"
