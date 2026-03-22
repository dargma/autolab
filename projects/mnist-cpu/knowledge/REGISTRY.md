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
- [exp-004] Ternary CNN [1,8,16]+FC128: 99.10% accuracy (16 epochs, STE+AdamW+cosine LR+label smoothing 0.05)
- [exp-004] Ternary weights: threshold δ=0.7*mean(|w|), alpha=mean(|w[|w|>δ]|), STE gradient pass-through
- [exp-004] Ternary model has 103,066 params vs 204,778 for tiny-cnn (50% fewer) with higher accuracy (99.10% vs 98.90%)
- [exp-005] ternary_v3.c C engine: 0.485ms avg, 0.004ms min, int16 fixed-point (Q8.8), zero multiplications in datapath
- [exp-005] C engine architecture: direct accumulation (no im2col), branchless add/sub, fused BN+ReLU, output-stationary loop order
- [exp-005] Ternary sparsity ~30-50% (zero weights skipped in accumulation) provides additional speedup
- [GOAL MET] ternary_cnn [8,16]+FC128: accuracy=99.10% (>98.9%), latency=0.485ms (<0.50ms) — both targets met simultaneously
## Hypotheses
<!-- Unverified claims and predictions -->
- ~~Very small CNNs (< 50K params) should easily meet 10fps~~ → CONFIRMED (exp-001: 26K@41ms)
- ~~Depthwise separable convs may offer better acc/latency tradeoff~~ → REJECTED at MNIST scale (exp-001)
- ~~A single FC layer might meet fps but accuracy ceiling ~95%~~ → REFINED: multi-layer FC reaches ~97.9%
- ~~Longer training (10-20 epochs) may push top candidates past 99%~~ → NOT on FashionMNIST (exp-002: 10ep, best=91.5%)
- Larger CNNs (32/64 channels) + batch norm could reach 93%+ on FashionMNIST within constraint
- On harder tasks, model capacity (params) matters more — diminishing returns less steep
- Latency anomaly: very small conv tensors (4ch) may have PyTorch overhead → needs profiling
- Knowledge distillation (teacher big-cnn → student ternary) may push accuracy past 99.2% with zero latency cost
- Zero-skipping in ternary convolutions (~30-50% zeros) could be exploited with explicit branch for further speedup

## Rejected Ideas
<!-- Ideas tried or analyzed and rejected + rejection rationale -->
- Depthwise separable for MNIST: fewer params but lower accuracy, no latency advantage (exp-001)
- Bit-packed ternary (v2, pos/neg bitmasks): slower than direct int8 (v3) due to im2col copy overhead
- PyTorch overhead makes sub-0.50ms inference impossible even for tiny models — custom C engine required

## Open Questions
<!-- What to explore next -->
- What is the Pareto frontier of accuracy vs latency on CPU for MNIST?
- At what model size does accuracy saturate on MNIST?
- Is quantization (int8) worth it at this scale?
- Why does minimal-cnn (26K) have 41ms latency while tiny-cnn (205K) has 0.89ms?
- Can batch norm + residual connections push past 99% within constraint?
- Can knowledge distillation push ternary accuracy past 99.2%?
- What is the latency floor for the C engine with zero-skipping enabled?
- Can the framework generalize to CIFAR-10 (32x32 RGB, 10 classes) without code changes?
- Is there a sweet spot for ternary threshold (currently 0.7*mean) that optimizes the accuracy-sparsity tradeoff?
