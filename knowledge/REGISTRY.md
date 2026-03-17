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

## Hypotheses
<!-- Unverified claims and predictions -->
- ~~Very small CNNs (< 50K params) should easily meet 10fps~~ → CONFIRMED (exp-001: 26K@41ms)
- ~~Depthwise separable convs may offer better acc/latency tradeoff~~ → REJECTED at MNIST scale (exp-001)
- ~~A single FC layer might meet fps but accuracy ceiling ~95%~~ → REFINED: multi-layer FC reaches ~97.9%
- Longer training (10-20 epochs) may push top candidates past 99%
- Larger CNNs (32/64 channels) + batch norm could reach 99.3%+ within constraint
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
