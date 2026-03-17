#!/usr/bin/env bash
# Exp-002: FashionMNIST architecture sweep (CPU, multi-process)
set -euo pipefail

USAGE=$(df -h . | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$USAGE" -ge 95 ]; then
  echo "⛔ DISK 95% — All operations halted."
  exit 1
fi

cd "$(dirname "$0")"
python3 sweep.py 2>&1 | tee results/train.log
echo "✅ Sweep complete. Results in results/"
