# Experiment Tracker

| # | Name | Status | Key Metric | Date | Notes |
|---|------|--------|------------|------|-------|
<!-- 🔵planned 🟡running 🟢done 🔴failed -->
| 001 | mnist-baseline-sweep | 🟢done | 98.90% (tiny-cnn) | 2026-03-17 | 8/8 passed, best=tiny-cnn 0.89ms |
| 002 | fashionmnist-sweep | 🟢done | 91.54% (big-cnn) | 2026-03-17 | 8/8 passed, best=big-cnn 0.40ms |
| 003 | creative-sweep | 🔴failed | OOM killed | 2026-03-21 | BN/Residual/SE + cosine LR — Colab OOM, only 1/4 workers survived |
| 004 | ternary-baseline | 🟢done | 99.10% (ep16) | 2026-03-22 | ternary_cnn [8,16]+FC128, 103K params, STE+AdamW+cosine, stopped ep16/20 |
| 005 | ternary-c-engine | 🟢done | 0.485ms avg | 2026-03-21 | ternary_v3.c: adder-only int16 fixed-point, zero-multiply inference |
| 006 | knowledge-distill | 🔵planned | — | 2026-03-22 | Teacher: big-cnn 98.90% → Student: ternary [8,16]+FC128, T=4, α=0.7 |
