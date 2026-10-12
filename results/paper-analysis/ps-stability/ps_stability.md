| Row | Selection accuracy (candidates) | Chosen | P(same choice), bootstrap of 200 | P(same), 50 questions | P(same), 25 questions | D under each candidate (RandOpt − SC@50) | Expected D, 200 / 50 |
|---|---|---|---|---|---|---|---|
| q15 | randopt 68.5 / plain 73.0 / boxed 80.5 | boxed | 0.996 | 0.891 | 0.702 | randopt -2.65 / plain +2.58 / boxed -2.96 | -2.93 / -2.53 |
| q3 | randopt 84.0 / plain 85.0 / boxed 90.0 | boxed | 0.955 | 0.730 | 0.520 | randopt -1.59 / plain +7.05 / boxed -0.38 | -0.19 / +0.53 |
| olmo | randopt 41.5 / plain 77.0 / boxed 79.5 | boxed | 0.775 | 0.611 | 0.523 | randopt +8.72 / plain -22.21 / boxed -23.65 | -23.33 / -23.09 |
| gqa | cot 53.5 / direct 66.0 / short 62.0 | direct | 0.996 | 0.941 | 0.823 | cot +3.63 / direct -1.21 / short -0.57 | -1.21 / -1.08 |

EXPLORATORY. No new generations; PS selection outputs and existing test-set SC@50 runs.
