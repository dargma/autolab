# Knowledge Registry

> This file is the single source of truth. Never delete entries — append and revise only.

## Established Facts
<!-- Only experimentally verified results -->
- MNIST: 28×28 grayscale, 10 classes, 60K train / 10K test
- 10fps on CPU = ≤100ms per inference (batch_size=1)
- [exp-001] Best: tiny-cnn [1,8,16]+FC256 → 98.90% acc, 0.89ms latency, 205K params
- [exp-001] All 8 candidates passed ≤100ms constraint; CNN top-3, FC bottom-3
- [exp-001] FC models plateau at 97.3–97.9% on MNIST (5 epochs, Adam lr=0.001)
- [exp-001] Depthwise separable (102K params) → 98.31%, not better than standard conv at this scale
- FashionMNIST: 28×28 grayscale, 10 classes (clothing), same size as MNIST but ~8-10% harder
- [exp-002] Best: big-cnn [1,16,32]+FC512 → 91.54% acc, 0.40ms latency, 813K params
- [exp-002] All 8 candidates passed ≤100ms; ranking shift: big-cnn > tiny-cnn on harder data
- [exp-002] MNIST winner (tiny-cnn) drops from 98.90% → 90.24% on FashionMNIST (-8.66pp)
- [exp-002] FC models: 87.6–88.7% on FashionMNIST (vs 97.3–97.9% on MNIST)
- [exp-002] Accuracy gap CNN vs FC widens on FashionMNIST: ~4pp (vs ~1pp on MNIST)
- [exp-002] More params helps more on harder data: big-cnn (813K) overtakes tiny-cnn (205K)

- [ralph] Goal met at iteration 0: tiny-cnn acc=0.9890, lat=0.89ms
## Hypotheses
<!-- Unverified claims and predictions -->
- ~~Very small CNNs (< 50K params) should easily meet 10fps~~ → CONFIRMED (exp-001: 26K@41ms)
- ~~Depthwise separable convs may offer better acc/latency tradeoff~~ → REJECTED at MNIST scale (exp-001)
- ~~A single FC layer might meet fps but accuracy ceiling ~95%~~ → REFINED: multi-layer FC reaches ~97.9%
- ~~Longer training (10-20 epochs) may push top candidates past 99%~~ → NOT on FashionMNIST (exp-002: 10ep, best=91.5%)
- Larger CNNs (32/64 channels) + batch norm could reach 93%+ on FashionMNIST within constraint
- On harder tasks, model capacity (params) matters more — diminishing returns less steep
- Latency anomaly: very small conv tensors (4ch) may have PyTorch overhead → needs profiling

## Rejected Ideas
<!-- Ideas tried or analyzed and rejected + rejection rationale -->
- Depthwise separable for MNIST: fewer params but lower accuracy, no latency advantage (exp-001)

## Open Questions
<!-- What to explore next -->
- What is the Pareto frontier of accuracy vs latency on CPU for MNIST?
- At what model size does accuracy saturate on MNIST?
- Is quantization (int8) worth it at this scale?
- Why does minimal-cnn (26K) have 41ms latency while tiny-cnn (205K) has 0.89ms?
- Can batch norm + residual connections push past 99% within constraint?
