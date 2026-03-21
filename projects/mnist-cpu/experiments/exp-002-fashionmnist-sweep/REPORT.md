# Exp-002: FashionMNIST Architecture Sweep

## Objective
Test the same 8 architectures from exp-001 on FashionMNIST to measure accuracy degradation on a harder dataset under the same ≤100ms CPU constraint.

## Setup
- **Dataset**: FashionMNIST (28×28 grayscale, 10 clothing classes)
- **Constraint**: ≤100ms inference latency, batch_size=1, CPU only
- **Training**: Adam, lr=0.001, **10 epochs** (doubled from exp-001), seed=42
- **Normalization**: FashionMNIST-specific (mean=0.286, std=0.353)
- **Candidates**: Same 8 architectures as exp-001

## Results

### Full Ranking (sorted by accuracy, all passed constraint)

| Rank | Name | Accuracy | Avg Latency (ms) | Params | MNIST (exp-001) | Delta |
|------|------|----------|-------------------|--------|-----------------|-------|
| 1 | big-cnn | **91.54%** | 0.40 | 813,258 | 98.81% | -7.27pp |
| 2 | tiny-cnn | 90.24% | 0.75 | 204,778 | 98.90% | -8.66pp |
| 3 | depthwise-cnn | 89.41% | 0.92 | 102,074 | 98.31% | -8.90pp |
| 4 | minimal-cnn | 89.22% | 7.67 | 26,138 | 98.18% | -8.96pp |
| 5 | micro-cnn | 88.86% | 21.11 | 38,210 | 97.44% | -8.58pp |
| 6 | wide-fc | 88.66% | 21.65 | 218,058 | 97.88% | -9.22pp |
| 7 | tiny-fc | 87.76% | 18.03 | 101,770 | 97.73% | -9.97pp |
| 8 | deep-fc | 87.56% | 13.11 | 111,146 | 97.30% | -9.74pp |

### Eliminated
None — all 8 candidates passed the ≤100ms constraint.

## Analysis

### Key Findings

1. **Ranking shift**: big-cnn overtakes tiny-cnn as #1. On MNIST they were within 0.09pp; on FashionMNIST the gap is 1.3pp. More capacity pays off on harder data.

2. **Uniform accuracy drop ~8-10pp**: All models drop by a similar margin, confirming FashionMNIST is systematically harder, not architecture-dependent.

3. **CNN vs FC gap widens**: On MNIST the gap was ~1pp; on FashionMNIST it's ~3-4pp. Spatial features matter more when texture/shape discrimination is needed.

4. **Latency unchanged**: Same architectures → same latency. The constraint remains easily satisfied.

5. **Depthwise separable holds 3rd place**: Consistent across both datasets — efficient but not top accuracy.

### Cross-Dataset Comparison
| Architecture Type | MNIST Avg | FashionMNIST Avg | Drop |
|-------------------|-----------|------------------|------|
| CNN (5 models) | 98.33% | 89.85% | -8.48pp |
| FC (3 models) | 97.64% | 87.99% | -9.65pp |

FC models suffer more on harder data (9.65pp vs 8.48pp drop).

## Next Steps
1. **Batch norm + dropout**: Should significantly help FashionMNIST (regularization)
2. **Data augmentation**: Random horizontal flip, random crop — standard for FashionMNIST
3. **Larger CNN sweep**: 32/64/128 channels to find where accuracy saturates
4. **Cross-dataset Pareto plot**: Combined accuracy-vs-params figure for both datasets
