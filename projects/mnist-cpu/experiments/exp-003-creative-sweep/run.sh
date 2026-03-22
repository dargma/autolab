#!/bin/bash
set -e

# Disk safety guard
USAGE=$(df -h . | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$USAGE" -ge 95 ]; then
  echo "DISK 95% — All operations halted. Cleanup required."
  exit 1
fi

echo "=== Exp-003: Creative MNIST Sweep ==="
echo "Goal: 98.9% accuracy + 0.50ms latency"
echo "Techniques: BatchNorm, Residual, SE-blocks, cosine LR, augmentation, JIT, label smoothing"
echo ""

cd "$(dirname "$0")/../.."
python3 -c "
import sys, yaml, json
sys.path.insert(0, '../..')
from autolab.sweep import SweepRunner

with open('experiments/exp-003-creative-sweep/config.yaml') as f:
    cfg = yaml.safe_load(f)

runner = SweepRunner('experiments/exp-003-creative-sweep/results')
results = runner.run(
    candidates=cfg['search']['candidates'],
    train_cfg=cfg['train'],
    constraint=cfg['constraint'],
    dataset_name=cfg['dataset'],
    n_workers=cfg['search']['parallel_workers'],
)

# Print summary
passing = [r for r in results if r['meets_constraint'] and r['status'] == 'ok']
passing.sort(key=lambda r: r['accuracy'], reverse=True)
print()
print('='*60)
print(f'TARGET: accuracy >= 98.9%, latency <= 0.50ms')
print('='*60)
for r in passing[:5]:
    target_met = r['accuracy'] >= 0.989 and r['avg_latency_ms'] <= 0.50
    flag = 'GOAL MET' if target_met else 'miss'
    print(f'  {r[\"name\"]:25s} acc={r[\"accuracy\"]:.4f} lat={r[\"avg_latency_ms\"]:.3f}ms [{flag}]')
" 2>&1 | tee experiments/exp-003-creative-sweep/results/train.log
