# Frozen complementarity study: complete derived tables

Accuracy values are percentages; deltas and confidence intervals are percentage points.
Seed 0 is primary. Other seeds and random controls are diagnostics only.
Inference calls count one full 300-question generation RPC per expert.
Expert-pass ratios are logical counts, not measured wall-clock speedups.

## Primary selection to held-out transfer

| Population | K | Standard selection | Greedy selection | Standard fresh | Greedy fresh | Selection delta | Fresh delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | 3 | 54.00 | 55.50 | 39.00 | 39.67 | +1.50 | +0.67 |
| development | 5 | 56.00 | 62.00 | 42.33 | 41.33 | +6.00 | -1.00 |
| development | 10 | 60.50 | 67.50 | 48.67 | 47.33 | +7.00 | -1.33 |
| development | 20 | 60.50 | 69.00 | 50.67 | 48.00 | +8.50 | -2.67 |
| development | 50 | 61.50 | 70.00 | 51.67 | not collected | +8.50 | not collected |
| validation_a | 3 | 53.50 | 55.50 | 36.33 | 38.67 | +2.00 | +2.33 |
| validation_a | 5 | 59.50 | 62.00 | 41.33 | 40.67 | +2.50 | -0.67 |
| validation_a | 10 | 61.00 | 66.50 | 48.00 | 46.00 | +5.50 | -2.00 |
| validation_a | 20 | 63.50 | 70.50 | 50.00 | 50.67 | +7.00 | +0.67 |
| validation_a | 50 | 61.50 | 70.00 | 52.00 | not collected | +8.50 | not collected |
| validation_b | 3 | 54.50 | 56.50 | 38.00 | 37.00 | +2.00 | -1.00 |
| validation_b | 5 | 58.00 | 62.00 | 43.67 | 44.00 | +4.00 | +0.33 |
| validation_b | 10 | 58.50 | 65.00 | 48.00 | 45.00 | +6.50 | -3.00 |
| validation_b | 20 | 60.50 | 68.00 | 49.67 | 49.00 | +7.50 | -0.67 |
| validation_b | 50 | 60.50 | 70.00 | 51.33 | not collected | +9.50 | not collected |

## Exact deployment costs and committee overlap

| Population | Method | K | Correct / 300 | Unique experts | Inference calls | Request sequences | Generated tokens | Capped % | Overlap with top-K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | standard | 3 | 117 | 3 | 3 | 900 | 294369 | 9.33 | 3 |
| development | greedy-0 | 3 | 119 | 3 | 3 | 900 | 292067 | 7.89 | 2 |
| development | standard | 5 | 127 | 5 | 5 | 1500 | 482149 | 8.73 | 5 |
| development | greedy-0 | 5 | 124 | 5 | 5 | 1500 | 474823 | 7.93 | 2 |
| development | standard | 10 | 146 | 10 | 10 | 3000 | 956544 | 8.50 | 10 |
| development | greedy-0 | 10 | 142 | 10 | 10 | 3000 | 958014 | 8.73 | 4 |
| development | standard | 20 | 152 | 20 | 20 | 6000 | 1894443 | 8.67 | 20 |
| development | greedy-0 | 20 | 144 | 20 | 20 | 6000 | 1943237 | 15.75 | 5 |
| development | standard | 50 | 155 | 50 | 50 | 15000 | 4689315 | 8.82 | 50 |
| validation_a | standard | 3 | 109 | 3 | 3 | 900 | 289481 | 10.11 | 3 |
| validation_a | greedy-0 | 3 | 116 | 3 | 3 | 900 | 287731 | 9.33 | 2 |
| validation_a | standard | 5 | 124 | 5 | 5 | 1500 | 475081 | 9.20 | 5 |
| validation_a | greedy-0 | 5 | 122 | 5 | 5 | 1500 | 481032 | 9.73 | 2 |
| validation_a | standard | 10 | 144 | 10 | 10 | 3000 | 941208 | 8.97 | 10 |
| validation_a | greedy-0 | 10 | 138 | 10 | 10 | 3000 | 998719 | 13.27 | 3 |
| validation_a | standard | 20 | 150 | 20 | 20 | 6000 | 1889772 | 9.13 | 20 |
| validation_a | greedy-0 | 20 | 152 | 20 | 20 | 6000 | 1968985 | 14.03 | 4 |
| validation_a | standard | 50 | 156 | 50 | 50 | 15000 | 4747187 | 9.29 | 50 |
| validation_b | standard | 3 | 114 | 3 | 3 | 900 | 283781 | 9.00 | 3 |
| validation_b | greedy-0 | 3 | 111 | 3 | 3 | 900 | 289149 | 9.89 | 2 |
| validation_b | standard | 5 | 131 | 5 | 5 | 1500 | 471843 | 9.00 | 5 |
| validation_b | greedy-0 | 5 | 132 | 5 | 5 | 1500 | 483880 | 9.20 | 2 |
| validation_b | standard | 10 | 144 | 10 | 10 | 3000 | 952098 | 9.37 | 10 |
| validation_b | greedy-0 | 10 | 135 | 10 | 10 | 3000 | 987477 | 17.87 | 4 |
| validation_b | standard | 20 | 149 | 20 | 20 | 6000 | 1906861 | 9.65 | 20 |
| validation_b | greedy-0 | 20 | 147 | 20 | 20 | 6000 | 1909782 | 13.40 | 5 |
| validation_b | standard | 50 | 154 | 50 | 50 | 15000 | 4753302 | 9.53 | 50 |

## Primary quality and diversity

| Population | Method | K | Mean individual selection | Mean individual fresh | Selection error correlation | Fresh error correlation | Fresh answer disagreement | Fresh both wrong | Mean vote margin | Vote tie % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | standard | 3 | 48.50 | 35.00 | 0.407 | 0.434 | 67.22 | 52.11 | 0.99 | 42.33 |
| development | greedy-0 | 3 | 48.00 | 35.78 | 0.440 | 0.492 | 63.89 | 52.56 | 1.09 | 41.00 |
| development | standard | 5 | 47.70 | 35.47 | 0.453 | 0.434 | 67.77 | 51.57 | 1.68 | 25.00 |
| development | greedy-0 | 5 | 45.40 | 33.53 | 0.390 | 0.449 | 67.93 | 54.13 | 1.67 | 28.00 |
| development | standard | 10 | 46.65 | 36.03 | 0.421 | 0.454 | 67.01 | 51.36 | 3.34 | 17.67 |
| development | greedy-0 | 10 | 43.00 | 33.00 | 0.367 | 0.426 | 70.37 | 54.28 | 3.04 | 21.67 |
| development | standard | 20 | 45.27 | 35.82 | 0.435 | 0.449 | 67.19 | 51.51 | 6.51 | 12.33 |
| development | greedy-0 | 20 | 32.50 | 26.35 | 0.259 | 0.307 | 81.29 | 60.13 | 4.51 | 16.33 |
| development | standard | 50 | 42.20 | 34.43 | 0.408 | 0.421 | 69.55 | 52.46 | 15.34 | 6.00 |
| validation_a | standard | 3 | 48.00 | 34.33 | 0.406 | 0.483 | 66.67 | 54.00 | 1.00 | 44.00 |
| validation_a | greedy-0 | 3 | 46.17 | 35.78 | 0.420 | 0.481 | 67.56 | 52.22 | 0.97 | 45.33 |
| validation_a | standard | 5 | 47.50 | 35.40 | 0.431 | 0.473 | 66.07 | 52.53 | 1.73 | 29.67 |
| validation_a | greedy-0 | 5 | 43.10 | 34.60 | 0.369 | 0.442 | 68.47 | 52.73 | 1.61 | 30.67 |
| validation_a | standard | 10 | 46.35 | 35.57 | 0.437 | 0.467 | 66.28 | 52.20 | 3.52 | 13.33 |
| validation_a | greedy-0 | 10 | 40.20 | 32.63 | 0.332 | 0.398 | 73.25 | 54.03 | 2.95 | 16.67 |
| validation_a | standard | 20 | 45.02 | 35.53 | 0.449 | 0.469 | 66.55 | 52.28 | 6.76 | 8.67 |
| validation_a | greedy-0 | 20 | 35.60 | 29.27 | 0.272 | 0.339 | 78.70 | 56.98 | 4.89 | 12.67 |
| validation_a | standard | 50 | 42.81 | 34.79 | 0.415 | 0.446 | 68.63 | 52.62 | 15.74 | 5.00 |
| validation_b | standard | 3 | 48.50 | 35.89 | 0.460 | 0.447 | 65.00 | 51.22 | 1.05 | 39.67 |
| validation_b | greedy-0 | 3 | 47.33 | 35.22 | 0.434 | 0.498 | 64.89 | 53.11 | 1.05 | 43.33 |
| validation_b | standard | 5 | 47.40 | 34.93 | 0.441 | 0.454 | 65.97 | 52.53 | 1.78 | 21.67 |
| validation_b | greedy-0 | 5 | 44.50 | 35.73 | 0.428 | 0.431 | 68.50 | 51.03 | 1.69 | 25.00 |
| validation_b | standard | 10 | 45.90 | 35.57 | 0.449 | 0.466 | 66.48 | 52.10 | 3.43 | 17.67 |
| validation_b | greedy-0 | 10 | 30.25 | 24.57 | 0.244 | 0.277 | 84.21 | 61.74 | 2.09 | 23.33 |
| validation_b | standard | 20 | 44.35 | 34.73 | 0.437 | 0.444 | 67.97 | 52.61 | 6.54 | 10.33 |
| validation_b | greedy-0 | 20 | 32.27 | 26.72 | 0.275 | 0.312 | 81.30 | 59.68 | 4.62 | 13.00 |
| validation_b | standard | 50 | 41.53 | 34.36 | 0.413 | 0.435 | 69.01 | 52.85 | 15.72 | 5.00 |

## Frozen tie seeds and matched controls

| Population | K | Greedy 0 | Greedy 1 | Greedy 2 | Random 101 | Random 102 | Random 103 | Singleton positions | Min score-group size | Max score-group size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 3 | 39.67 | 39.67 | 39.67 | 39.67 | 39.67 | 39.67 | 2 | 1 | 4 |
| development | 5 | 41.33 | 41.33 | 41.33 | 43.00 | 42.67 | 42.67 | 2 | 1 | 5 |
| development | 10 | 47.33 | 47.33 | 47.33 | 47.33 | 49.00 | 49.00 | 3 | 1 | 8 |
| development | 20 | 48.00 | 48.00 | 48.00 | 49.00 | 49.33 | 49.33 | 3 | 1 | 8 |
| validation_a | 3 | 38.67 | 38.67 | 38.67 | 41.33 | 38.67 | 36.67 | 1 | 1 | 7 |
| validation_a | 5 | 40.67 | 40.67 | 40.67 | 43.67 | 43.67 | 39.67 | 1 | 1 | 7 |
| validation_a | 10 | 46.00 | 48.00 | 48.00 | 48.33 | 46.67 | 47.67 | 1 | 1 | 7 |
| validation_a | 20 | 50.67 | 49.67 | 49.67 | 48.33 | 49.00 | 50.33 | 3 | 1 | 10 |
| validation_b | 3 | 37.00 | 37.00 | 37.00 | 37.33 | 37.33 | 37.00 | 2 | 1 | 4 |
| validation_b | 5 | 44.00 | 44.00 | 44.00 | 40.67 | 40.00 | 42.33 | 2 | 1 | 4 |
| validation_b | 10 | 45.00 | 45.00 | 45.00 | 42.00 | 41.33 | 44.67 | 2 | 1 | 4 |
| validation_b | 20 | 49.00 | 49.00 | 49.00 | 48.67 | 45.33 | 49.33 | 3 | 1 | 11 |

## Exact-quality random diversity controls

| Population | K | Seed | Selection accuracy | Fresh accuracy | Primary greedy minus random | Fresh error correlation | Fresh answer disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | 3 | 101 | 55.50 | 39.67 | +0.00 | 0.492 | 63.89 |
| development | 3 | 102 | 55.50 | 39.67 | +0.00 | 0.492 | 63.89 |
| development | 3 | 103 | 55.50 | 39.67 | +0.00 | 0.492 | 63.89 |
| development | 5 | 101 | 58.50 | 43.00 | -1.67 | 0.430 | 69.07 |
| development | 5 | 102 | 62.00 | 42.67 | -1.33 | 0.423 | 69.73 |
| development | 5 | 103 | 62.00 | 42.67 | -1.33 | 0.423 | 69.73 |
| development | 10 | 101 | 61.50 | 47.33 | +0.00 | 0.429 | 69.51 |
| development | 10 | 102 | 62.50 | 49.00 | -1.67 | 0.402 | 71.62 |
| development | 10 | 103 | 60.50 | 49.00 | -1.67 | 0.412 | 70.95 |
| development | 20 | 101 | 58.00 | 49.00 | -1.00 | 0.299 | 81.55 |
| development | 20 | 102 | 63.00 | 49.33 | -1.33 | 0.295 | 81.61 |
| development | 20 | 103 | 60.00 | 49.33 | -1.33 | 0.296 | 81.41 |
| validation_a | 3 | 101 | 53.00 | 41.33 | -2.67 | 0.482 | 67.44 |
| validation_a | 3 | 102 | 55.50 | 38.67 | +0.00 | 0.481 | 67.56 |
| validation_a | 3 | 103 | 51.00 | 36.67 | +2.00 | 0.483 | 66.33 |
| validation_a | 5 | 101 | 59.00 | 43.67 | -3.00 | 0.424 | 71.90 |
| validation_a | 5 | 102 | 57.00 | 43.67 | -3.00 | 0.445 | 70.73 |
| validation_a | 5 | 103 | 55.00 | 39.67 | +1.00 | 0.448 | 68.73 |
| validation_a | 10 | 101 | 62.50 | 48.33 | -2.33 | 0.411 | 73.56 |
| validation_a | 10 | 102 | 62.00 | 46.67 | -0.67 | 0.428 | 73.52 |
| validation_a | 10 | 103 | 59.50 | 47.67 | -1.67 | 0.416 | 72.39 |
| validation_a | 20 | 101 | 63.50 | 48.33 | +2.33 | 0.368 | 77.85 |
| validation_a | 20 | 102 | 61.50 | 49.00 | +1.67 | 0.349 | 78.33 |
| validation_a | 20 | 103 | 61.50 | 50.33 | +0.33 | 0.357 | 77.57 |
| validation_b | 3 | 101 | 52.50 | 37.33 | -0.33 | 0.559 | 60.89 |
| validation_b | 3 | 102 | 52.50 | 37.33 | -0.33 | 0.559 | 60.89 |
| validation_b | 3 | 103 | 55.00 | 37.00 | +0.00 | 0.446 | 68.11 |
| validation_b | 5 | 101 | 55.50 | 40.67 | +3.33 | 0.488 | 64.10 |
| validation_b | 5 | 102 | 55.00 | 40.00 | +4.00 | 0.518 | 63.70 |
| validation_b | 5 | 103 | 58.50 | 42.33 | +1.67 | 0.433 | 68.20 |
| validation_b | 10 | 101 | 55.50 | 42.00 | +3.00 | 0.319 | 82.39 |
| validation_b | 10 | 102 | 53.50 | 41.33 | +3.67 | 0.314 | 82.32 |
| validation_b | 10 | 103 | 59.00 | 44.67 | +0.33 | 0.276 | 84.10 |
| validation_b | 20 | 101 | 59.00 | 48.67 | +0.33 | 0.344 | 79.63 |
| validation_b | 20 | 102 | 58.00 | 45.33 | +3.67 | 0.334 | 79.71 |
| validation_b | 20 | 103 | 58.00 | 49.33 | -0.33 | 0.314 | 80.62 |

## Efficiency and ordinary-small-K checks

| Population | Comparison | Selection delta | Fresh delta | Paired descriptive 95% interval | Expert-pass ratio | Pass reduction % | Token reduction % | Improved questions | Worsened questions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | greedy-0-k5_vs_standard-k20 | +1.50 | -9.33 | [-14.00, -4.67] | 4 | 75.00 | 74.94 | 11 | 39 |
| development | standard-k5_vs_standard-k20 | -4.50 | -8.33 | [-12.33, -4.33] | 4 | 75.00 | 74.55 | 7 | 32 |
| development | greedy-0-k5_vs_standard-k50 | +0.50 | -10.33 | [-14.67, -6.00] | 10 | 90.00 | 89.87 | 9 | 40 |
| development | standard-k5_vs_standard-k50 | -5.50 | -9.33 | [-13.67, -5.00] | 10 | 90.00 | 89.72 | 9 | 37 |
| development | greedy-0-k10_vs_standard-k20 | +7.00 | -3.33 | [-7.01, +0.33] | 2 | 50.00 | 49.43 | 12 | 22 |
| development | standard-k10_vs_standard-k20 | +0.00 | -2.00 | [-5.67, +1.33] | 2 | 50.00 | 49.51 | 11 | 17 |
| development | greedy-0-k10_vs_standard-k50 | +6.00 | -4.33 | [-8.33, -0.33] | 5 | 80.00 | 79.57 | 11 | 24 |
| development | standard-k10_vs_standard-k50 | -1.00 | -3.00 | [-6.67, +0.33] | 5 | 80.00 | 79.60 | 10 | 19 |
| validation_a | greedy-0-k5_vs_standard-k20 | -1.50 | -9.33 | [-14.00, -4.67] | 4 | 75.00 | 74.55 | 12 | 40 |
| validation_a | standard-k5_vs_standard-k20 | -4.00 | -8.67 | [-13.00, -4.33] | 4 | 75.00 | 74.86 | 10 | 36 |
| validation_a | greedy-0-k5_vs_standard-k50 | +0.50 | -11.33 | [-15.67, -7.33] | 10 | 90.00 | 89.87 | 6 | 40 |
| validation_a | standard-k5_vs_standard-k50 | -2.00 | -10.67 | [-15.00, -6.33] | 10 | 90.00 | 89.99 | 7 | 39 |
| validation_a | greedy-0-k10_vs_standard-k20 | +3.00 | -4.00 | [-8.33, +0.33] | 2 | 50.00 | 47.15 | 18 | 30 |
| validation_a | standard-k10_vs_standard-k20 | -2.50 | -2.00 | [-4.67, +0.67] | 2 | 50.00 | 50.19 | 6 | 12 |
| validation_a | greedy-0-k10_vs_standard-k50 | +5.00 | -6.00 | [-10.00, -2.33] | 5 | 80.00 | 78.96 | 10 | 28 |
| validation_a | standard-k10_vs_standard-k50 | -0.50 | -4.00 | [-7.33, -1.00] | 5 | 80.00 | 80.17 | 6 | 18 |
| validation_b | greedy-0-k5_vs_standard-k20 | +1.50 | -5.67 | [-9.67, -1.67] | 4 | 75.00 | 74.62 | 12 | 29 |
| validation_b | standard-k5_vs_standard-k20 | -2.50 | -6.00 | [-10.00, -2.00] | 4 | 75.00 | 75.26 | 10 | 28 |
| validation_b | greedy-0-k5_vs_standard-k50 | +1.50 | -7.33 | [-11.67, -3.33] | 10 | 90.00 | 89.82 | 10 | 32 |
| validation_b | standard-k5_vs_standard-k50 | -2.50 | -7.67 | [-11.67, -3.67] | 10 | 90.00 | 90.07 | 9 | 32 |
| validation_b | greedy-0-k10_vs_standard-k20 | +4.50 | -4.67 | [-9.00, -0.33] | 2 | 50.00 | 48.21 | 16 | 30 |
| validation_b | standard-k10_vs_standard-k20 | -2.00 | -1.67 | [-4.33, +1.00] | 2 | 50.00 | 50.07 | 6 | 11 |
| validation_b | greedy-0-k10_vs_standard-k50 | +4.50 | -6.33 | [-10.67, -2.33] | 5 | 80.00 | 79.23 | 12 | 31 |
| validation_b | standard-k10_vs_standard-k50 | -2.00 | -3.33 | [-6.67, +0.00] | 5 | 80.00 | 79.97 | 7 | 17 |

## Frozen gate

```json
{
  "pass": false,
  "decision": "NO-GO_close_direction",
  "efficiency": [
    {
      "complementary_k": 5,
      "standard_k": 20,
      "deltas": {
        "validation_a": -0.09333333333333332,
        "validation_b": -0.05666666666666664
      },
      "pass": false
    },
    {
      "complementary_k": 5,
      "standard_k": 50,
      "deltas": {
        "validation_a": -0.11333333333333334,
        "validation_b": -0.0733333333333333
      },
      "pass": false
    },
    {
      "complementary_k": 10,
      "standard_k": 20,
      "deltas": {
        "validation_a": -0.03999999999999998,
        "validation_b": -0.046666666666666634
      },
      "pass": false
    },
    {
      "complementary_k": 10,
      "standard_k": 50,
      "deltas": {
        "validation_a": -0.06,
        "validation_b": -0.0633333333333333
      },
      "pass": false
    }
  ],
  "accuracy": [
    {
      "k": 3,
      "deltas": {
        "validation_a": 0.023333333333333317,
        "validation_b": -0.010000000000000009
      },
      "pass": false
    },
    {
      "k": 5,
      "deltas": {
        "validation_a": -0.006666666666666654,
        "validation_b": 0.003333333333333355
      },
      "pass": false
    },
    {
      "k": 10,
      "deltas": {
        "validation_a": -0.019999999999999962,
        "validation_b": -0.02999999999999997
      },
      "pass": false
    },
    {
      "k": 20,
      "deltas": {
        "validation_a": 0.00666666666666671,
        "validation_b": -0.006666666666666654
      },
      "pass": false
    }
  ],
  "primary_seed": 0,
  "diagnostic_seed_gates_not_eligible": {
    "1": {
      "pass": false,
      "decision": "NO-GO_close_direction",
      "efficiency": [
        {
          "complementary_k": 5,
          "standard_k": 20,
          "deltas": {
            "validation_a": -0.09333333333333332,
            "validation_b": -0.05666666666666664
          },
          "pass": false
        },
        {
          "complementary_k": 5,
          "standard_k": 50,
          "deltas": {
            "validation_a": -0.11333333333333334,
            "validation_b": -0.0733333333333333
          },
          "pass": false
        },
        {
          "complementary_k": 10,
          "standard_k": 20,
          "deltas": {
            "validation_a": -0.020000000000000018,
            "validation_b": -0.046666666666666634
          },
          "pass": false
        },
        {
          "complementary_k": 10,
          "standard_k": 50,
          "deltas": {
            "validation_a": -0.040000000000000036,
            "validation_b": -0.0633333333333333
          },
          "pass": false
        }
      ],
      "accuracy": [
        {
          "k": 3,
          "deltas": {
            "validation_a": 0.023333333333333317,
            "validation_b": -0.010000000000000009
          },
          "pass": false
        },
        {
          "k": 5,
          "deltas": {
            "validation_a": -0.006666666666666654,
            "validation_b": 0.003333333333333355
          },
          "pass": false
        },
        {
          "k": 10,
          "deltas": {
            "validation_a": 0.0,
            "validation_b": -0.02999999999999997
          },
          "pass": false
        },
        {
          "k": 20,
          "deltas": {
            "validation_a": -0.003333333333333355,
            "validation_b": -0.006666666666666654
          },
          "pass": false
        }
      ]
    },
    "2": {
      "pass": false,
      "decision": "NO-GO_close_direction",
      "efficiency": [
        {
          "complementary_k": 5,
          "standard_k": 20,
          "deltas": {
            "validation_a": -0.09333333333333332,
            "validation_b": -0.05666666666666664
          },
          "pass": false
        },
        {
          "complementary_k": 5,
          "standard_k": 50,
          "deltas": {
            "validation_a": -0.11333333333333334,
            "validation_b": -0.0733333333333333
          },
          "pass": false
        },
        {
          "complementary_k": 10,
          "standard_k": 20,
          "deltas": {
            "validation_a": -0.020000000000000018,
            "validation_b": -0.046666666666666634
          },
          "pass": false
        },
        {
          "complementary_k": 10,
          "standard_k": 50,
          "deltas": {
            "validation_a": -0.040000000000000036,
            "validation_b": -0.0633333333333333
          },
          "pass": false
        }
      ],
      "accuracy": [
        {
          "k": 3,
          "deltas": {
            "validation_a": 0.023333333333333317,
            "validation_b": -0.010000000000000009
          },
          "pass": false
        },
        {
          "k": 5,
          "deltas": {
            "validation_a": -0.006666666666666654,
            "validation_b": 0.003333333333333355
          },
          "pass": false
        },
        {
          "k": 10,
          "deltas": {
            "validation_a": 0.0,
            "validation_b": -0.02999999999999997
          },
          "pass": false
        },
        {
          "k": 20,
          "deltas": {
            "validation_a": -0.003333333333333355,
            "validation_b": -0.006666666666666654
          },
          "pass": false
        }
      ]
    }
  }
}
```

Per-question votes/margins, full individual score distributions, and every pair of expert errors are preserved in `analysis/committee-details.json.gz`.
