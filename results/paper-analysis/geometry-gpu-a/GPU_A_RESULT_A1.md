# GPU-A result under amendment A1: **NO-GO for GPU-B** (F1 falsified)

Lock: `plan_lock_A1.md` + unchanged parts of `plan_lock.md`. All gates and thresholds below were fixed before S2c ran.
Run-2 spend ≈ $4.56 of the $5.00 cap (session 1 ≈ $2.98; S2c session ≈ $1.58; pods stopped).

## Gates passed (engineering validity)
- V0 score definition, S2a prompt tokens, S2b exact candidate bytes (642/642 tensors SHA-identical).
- S2c ran in full: 961 per-example folded gradients; predictions for 543 RERANK / 61 TEST / 1000 SEARCH candidates;
  set projections for all 5000. Train-mode (checkpointed) forward reproduced the stored fp32 base contrast to ≤ 7.6e-6.

## Locked falsifiers (pre-registered)
| Test | Result | Threshold | Verdict |
|---|---|---|---|
| **F1 (primary)**: first-order prediction vs HF-internal measured contrast shift, 13 candidates × 761 examples | pooled r = **0.137**, slope 0.027, residual/signal energy 25.1; per-candidate r 0.06–0.32 | r < 0.3 falsifies; ≥ 0.6 supports | **Falsified** |
| F2: winner's changed answers predicted to flip | 45/85 = 0.53 | < 0.5 falsifies | not falsified (weak; 90 false flips on unchanged) |
| F3-RERANK: predicted vs observed accuracy gain across 543 candidates | r = 0.376 (Spearman 0.357) | < 0.3 falsifies; ≥ 0.5 supports | inconclusive |
| S2d: group partials vs measured single-group insertions (83 LOCALIZATION ex.) | pooled r = **0.239** | < 0.5 falsifies | **Falsified** |

Reported, not gating: F3-SEARCH r = 0.02; F3-TEST r = 0.30. Winner percentile of the set projection:
SEARCH 70.0, RERANK 96.6, TEST 0.8. Cosines of set folded gradients: SEARCH·RERANK 0.159, SEARCH·TEST 0.081,
RERANK·TEST 0.031.

**Decision (per the agreed rule "if S2c fails, stop"):** H★ — the whole-model first-order prediction
t̂ = σ⟨fold(∇s), ε⟩ — is falsified at the candidate's actual radius. GPU-B is not earned. The phrase "implicit gradient
step" must not be used as a claim.

## Post-hoc diagnosis (NOT pre-registered; hypothesis-generating only)
Per-group first-order partials vs measured single-group insertion shifts (vLLM, 83 LOCALIZATION examples, winner):

| group | sd predicted | sd measured | r | slope |
|---|---|---|---|---|
| vision | 7.12 | 1.52 | 0.05 | 0.01 |
| embed | 0.10 | 0.19 | 0.23 | 0.41 |
| lm_q1 | 1.39 | 1.62 | 0.71 | 0.83 |
| lm_q2 | 1.35 | 1.23 | 0.90 | 0.82 |
| lm_q3 | 1.15 | 1.08 | 0.94 | 0.89 |
| lm_q4 | 0.50 | 0.52 | 0.96 | 0.99 |
| final_norm_head | 0.22 | 0.26 | 0.86 | 1.03 |

Reading: the language-model perturbation is close to linear at σ = 0.002 (r 0.71–0.96, slopes ≈ 0.8–1.0), so the
gradient machinery is correct. The vision-tower perturbation is strongly nonlinear: its first-order term is ~5× too
large and uncorrelated with its actual effect, and it dominates the whole-model prediction. The vision effect itself is
not small (sd 1.5 nats, comparable to an LM quarter), so dropping vision would not rescue H★ as stated.

Any follow-up built on this (e.g. "LM-side shifts are first-order predictable; vision acts nonlinearly") is a new
hypothesis that needs its own lock before any new GPU work.
