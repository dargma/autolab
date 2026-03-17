# Exp-001: MNIST Baseline Architecture Sweep

## Objective
Find the optimal MNIST classifier under a 10fps (≤100ms) CPU latency constraint via parallel architecture search.

## Setup
- **Task**: MNIST 28x28 grayscale, 10 classes, 60K train / 10K test
- **Constraint**: ≤100ms inference latency, batch_size=1, CPU only
- **Training**: Adam, lr=0.001, 5 epochs, seed=42
- **Candidates**: 8 architectures (3 FC variants, 4 CNN variants, 1 depthwise separable)
- **Execution**: multiprocessing.Pool with 8 workers, apply_async + live CSV streaming

## Results

### Full Ranking (sorted by accuracy, all passed constraint)

| Rank | Name | Accuracy | Avg Latency (ms) | P99 Latency (ms) | Params | Pass |
|------|------|----------|-------------------|-------------------|--------|------|
| 1 | tiny-cnn | **98.90%** | 0.89 | 1.22 | 204,778 | Yes |
| 2 | big-cnn | 98.81% | 0.40 | 0.56 | 813,258 | Yes |
| 3 | depthwise-cnn | 98.31% | 1.22 | 1.67 | 102,074 | Yes |
| 4 | minimal-cnn | 98.18% | 41.40 | 56.84 | 26,138 | Yes |
| 5 | wide-fc | 97.88% | 17.56 | 24.11 | 218,058 | Yes |
| 6 | tiny-fc | 97.73% | 22.61 | 31.04 | 101,770 | Yes |
| 7 | micro-cnn | 97.44% | 42.73 | 58.63 | 38,210 | Yes |
| 8 | deep-fc | 97.30% | 16.55 | 22.72 | 111,146 | Yes |

### Eliminated
None — all 8 candidates passed the ≤100ms constraint.

## Analysis

### Key Findings
1. **CNN architectures dominate**: Top 3 are all CNNs. Convolutions exploit spatial structure that FC layers miss.
2. **tiny-cnn wins** (98.90%, 0.89ms): Best accuracy with sub-millisecond latency. The [1→8→16] channel config with 256-dim FC head hits the sweet spot.
3. **big-cnn is fastest** (0.40ms) but 4x more params for only 0.09% less accuracy — not efficient.
4. **Depthwise separable** (98.31%): Fewer params than tiny-cnn (102K vs 205K) but 0.6% lower accuracy. The parameter savings don't help at this scale.
5. **FC models plateau ~97.3–97.9%**: Confirms hypothesis that FC-only hits a ~95–98% ceiling on MNIST.
6. **Latency anomaly**: minimal-cnn (26K params) has 41ms latency vs tiny-cnn (205K params) at 0.89ms. Likely due to PyTorch overhead on very small tensors or suboptimal memory access patterns with 4-channel convolutions.

### Hypothesis Validation
- **"Very small CNNs < 50K params easily meet 10fps"** → CONFIRMED (minimal-cnn: 26K params, 41ms ≪ 100ms)
- **"Depthwise separable offers better acc/latency tradeoff"** → PARTIALLY REJECTED at this scale. Tiny-cnn with standard convs wins on both accuracy and latency.
- **"Single FC layer accuracy ceiling ~95%"** → REFINED: Multi-layer FC reaches ~97.9% but still trails CNNs by ~1%.

## Next Steps
1. **Pareto analysis**: Plot accuracy vs params and accuracy vs latency to visualize the frontier
2. **Extended search**: Try larger CNNs (32/64 channels), residual connections, batch norm
3. **Quantization test**: Check if int8 quantization benefits at this model scale
4. **Longer training**: 5 epochs may be underfitting — try 10–20 epochs for top candidates
