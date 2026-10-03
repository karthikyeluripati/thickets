# Independent raw-token reanalysis

Validated 333 checksummed files and all 300 candidate identities.
Recomputed all per-prompt metrics, simulations, aggregates and the frozen gate.

Quantiles below are pooled descriptive prompt/candidate distributions, not iid estimates.

| Workload | Sigma | LCP p10 / median / p90 | Mean LCP | Mean LCP fraction | Exact observed outputs | Immediate divergence | Candidate capped |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k32 | 0.0005 | 1 / 17 / 64 | 27.73 | 9.28% | 0.72% | 7.03% | 9.34% |
| gsm8k32 | 0.001 | 0 / 3 / 30 | 11.11 | 3.70% | 0.00% | 14.09% | 12.44% |
| gsm8k32 | 0.002 | 0 / 1 / 13 | 3.78 | 1.27% | 0.00% | 35.03% | 27.53% |
| natural8 | 0.0005 | 0 / 20 / 35 | 20.05 | 69.88% | 58.50% | 11.50% | 0.00% |
| natural8 | 0.001 | 0 / 15 / 31 | 14.76 | 53.90% | 38.62% | 17.38% | 0.38% |
| natural8 | 0.002 | 0 / 3 / 20 | 7.00 | 26.71% | 10.62% | 22.00% | 5.12% |

| Workload | Sigma | Initial k=2 / 4 / 8 / 16 match (all pairs) | Ideal round reduction k=2 / 4 / 8 / 16 |
|---|---:|---:|---:|
| gsm8k32 | 0.0005 | 75.72% / 68.97% / 65.62% / 53.81% | 4.33% / 6.48% / 7.54% / 8.08% |
| gsm8k32 | 0.001 | 59.09% / 49.69% / 41.28% / 28.41% | 1.81% / 2.67% / 3.10% / 3.31% |
| gsm8k32 | 0.002 | 34.19% / 24.72% / 18.47% / 7.16% | 0.63% / 0.91% / 1.03% / 1.10% |
| natural8 | 0.0005 | 87.88% / 83.25% / 78.75% / 61.88% | 29.36% / 44.26% / 51.37% / 55.10% |
| natural8 | 0.001 | 80.50% / 71.38% / 64.50% / 41.38% | 21.58% / 32.49% / 37.71% / 40.41% |
| natural8 | 0.002 | 70.12% / 47.38% / 32.00% / 12.50% | 10.09% / 15.11% / 17.39% / 18.43% |

Initial acceptance is distinct from conditional later blocks attempted before rejection.
The complete distributions, eligible denominators, conditional block acceptance and wasted
verification positions remain in the immutable summary and per-candidate raw artifacts.
No rejoining after rejection is assumed. These simulated reductions are not measured speedups.

Frozen gate: **negative**.
