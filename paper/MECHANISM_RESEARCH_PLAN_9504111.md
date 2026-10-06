# Mechanistic investigation of candidate 9504111: research plan

Status: plan only. No new GPU work has run.

- **Branch:** `research/perspective-mechanism-plan`, from local `bf085b7`
  (local only).
- **Basis:** the committed raw artifacts, not prose summaries.

The guiding rule is **localize → intervene → claim only after causal
validation**. One property of the perturbation family, verified on the H100
in the forensic audit, changes what "localization" can mean here. It is
stated first because it shapes every stage.

> **Structural fact (forensic Phase G, verified on the H100).** The pinned
> RandOpt worker calls `manual_seed(seed)` for *every* tensor and draws
> `randn` in BF16 on CUDA.
>
> - Equal-shape tensors receive bit-identical noise.
> - Smaller draws are exact prefixes of larger ones.
>
> So a candidate's whole 8.77B-parameter perturbation is the first 622.3M
> values of **one Gaussian stream** ε_seed (622.3M is the numel of the
> embedding and `lm_head`), sliced and reshaped into each of the 642 tensors.
> The 642 tensors fall into 19 shape groups. For example, all 36 language
> layers get the *same* q/k/v, o, gate/up and down noise matrices, and all 27
> vision blocks share theirs.
>
> **Consequences:**
>
> 1. No candidate is spatially localized, and all are identical in
>    distribution by location and magnitude. The measured RMS of δ is 0.0020 in
>    every group.
> 2. Candidates differ only in the *direction* of ε within a ≤ 622M-dim space.
> 3. Every perturbation is **depth-coherent**: the same matrix is added at
>    every layer.

## 1. What we actually know

**Reproduced and frozen:**

| Item | Value |
|---|---|
| Candidate | seed 9504111, σ = 0.002, all 642 vLLM tensors incl. vision; BF16 in-place add on the pinned base (`0c351dd`); study fingerprint `7f0bdb3b…`, engine hash `7d7ef38b…` |
| Phases | SEARCH200 (selection), RERANK200 (second-stage selection), TEST561 (official test, frozen and untouched until the end) |
| SEARCH | 72 → 77 (+2.5 pp; entered the top 50 through a 33-way tie) |
| RERANK | 79 → 95 (+8.0 pp; 20 repairs, 4 regressions) |
| TEST | 259 → 244 (−2.67 pp; 23 repairs, 38 regressions) |
| Transitions RERANK / TEST | both correct 75 / 221; both wrong, same answer 89 / 259; wrong → different wrong 12 / 20 |

**Behavioral facts** (post hoc, from the causal-diagnostic, margin, selection
and random-control reports):

1. **Close calls.** Changed answers were close contests under the base:
   median base top-2 gap 0.4–1.6 nats, against 4–5 for unchanged answers.
2. **No coarse localization.** Copying candidate values into one of 7
   parameter groups (vision, embed, 4 language quarters, norm + head)
   reproduces or reverts only 0–36% of answer changes. Embed and head
   contribute about 0.
3. **Additive effects.** Group effects add up in score space (localization
   only): the sum of the 7 single-group insertions predicts the candidate's
   score vectors with 14% residual energy and gets 45 of 59 changed answers.
   That is consistent with a near-first-order response.
4. **Not opportunity mix.** Matching on base correctness and margin leaves the
   RERANK–TEST gap almost unchanged (−10.6 pp, composition only −0.7). Scores
   move toward correct on RERANK (+0.40 nats, standardized) and away on TEST
   (−0.32).
5. **A shared tilt.** 12 random σ = 0.002 controls, chosen by an
   outcome-independent rule, favour RERANK over TEST by −1.65 pp on average.
   Their score shifts *flatten* the base (BASE-wrong margins rise,
   BASE-correct margins fall).
6. **An atypical winner.** 9504111 instead protects BASE-correct answers. Its
   score-change vector aligns poorly with the control mean (cosine about 0.2,
   against about 0.5 between controls). It has the worst TEST of 61 measured
   candidates.
7. **A real but unidentified RERANK advantage.** It survives held-out RERANK
   halves (+7.2 pp, own cross-fit). Population reliability (0.30–0.53) shrinks
   it to about +3.7 to +4.5. Selection optimism is 0.8–4.3 pp.
8. **No letter bias.** A fixed letter-offset model explains 25% of changes.

## 2. What we do NOT know

- **Which internal computation** changes, if any specific one does.
- **Whether a near-first-order model** (Δscore ≈ gradient · δ) explains the
  per-example and per-candidate effects quantitatively. The additivity result
  only *suggests* this.
- **Whether the mirage follows from geometry:** whether the set-level
  "improvement directions" of SEARCH, RERANK and TEST are misaligned, so that
  selecting on one set picks perturbations that do not transfer.
- **The source of the shared flattening tilt.** It could be a second-order
  (curvature) effect of isotropic noise, and if so, why it differs by set.
- **What depth coherence contributes.** Does the replicated-across-layers
  structure amplify effects compared with an independent per-layer
  perturbation of the same norm?
- **Whether anything generalizes** beyond this model and task.

## 3. Strongest current scientific hypothesis

**H★ (selection as implicit gradient alignment).** At σ = 0.002 a candidate's
answer-score changes are, to first order,
t_k[j] ≈ ⟨∇_θ s_j(θ₀), δ_k⟩, where s_j = correct-answer score minus the base's
strongest wrong option.

δ_k is a slice-replicated Gaussian stream. So for each set D,
⟨Σ_{j∈D} ∇s_j, δ_k⟩ = ⟨F_D, ε_k⟩, where F_D ("folded gradient") sums every
tensor's gradient onto the shared stream coordinates.

**Consequences:**

- Selecting the maximum on a set selects ε aligned with F_D.
- For isotropic ε, the expected transfer to another set is linear in their
  alignment: E[⟨F_T, ε⟩ | ⟨F_R, ε⟩ = a] = a · ⟨F_R, F_T⟩ / ‖F_R‖².
- **The expert mirage is predicted when F_RERANK and F_TEST are weakly
  aligned or anti-aligned.**
- The shared flattening tilt is the second-order term ½σ² · tr(H_j) (H_j the
  Hessian of s_j), averaged differently over the two sets.
- 9504111 would be an extreme draw along F_RERANK, not a special circuit.

This is a mechanism at the level of **parameter-space geometry** (where the
selected direction points relative to each evaluation set's gradient), not a
circuit. It is falsifiable, quantitative, and cheap to test.

## 4. Alternative hypotheses (ranked by current evidence)

| # | Hypothesis | Current evidence | Prior |
|---|---|---|---|
| H★ | First-order gradient alignment plus selection; the mirage comes from cross-set misalignment | Additivity; close-margin dependence; population behaviour | High |
| H-coh | Depth coherence (identical noise in every layer) is what makes σ = 0.002 effects large and structured | Structural fact; group effects split roughly evenly across depth quarters | Medium; testable with one intervention |
| H-flat | The shared tilt is curvature-driven flattening, and set differences come from item curvature | Controls' flattening profile | Medium |
| H-vis | A localized change in visual or spatial processing (vision tower, DeepStack injection) drives the RERANK gain | Vision contributes to TEST regressions (+0.21 share) but *opposes* RERANK repairs (−0.11); single-group S ≤ 0.31 | Low |
| H-head | Answer-letter / readout bias | Head and embed about 0; offset model 25% | Very low |
| H-nonlin | Thresholded, interacting nonlinear effects; no compact account | Additivity residual 14%; 2 RERANK regressions missed | Low–medium; the main NO-GO route |
| H-acc | 9504111 is an accident: the extreme of a near-Gaussian score distribution with nothing to explain beyond H★ | Compatible with H★ | Part of H★ |

## 5. Stage 0: phenomenon reconstruction (CPU, $0; essentially done)

**Hypothesis → Experiment → Measurement → Falsifier → Cost → Interpretation:**

- **Hypothesis:** the frozen artifacts fully define the phenomenon.
- **Experiment:** compile a machine-readable spec (`phenomenon_spec.json`)
  from the raw outputs and scores, not from prose. It contains candidate
  identity, the RNG recipe, the 19 shape groups, the per-example transitions,
  base margins and A–D scores, and control profiles.
- **Measurement:** counts reproduce (already verified in four independent
  reports).
- **Falsifier:** any count mismatch → stop.
- **Cost:** $0.
- **Interpretation:** the baseline for every later test.

## 6. Stage 1: static localization (CPU + 2 GPU-minutes)

**S1a. Structural decomposition.** Already answered by construction. There is
no location- or magnitude-based signature to find: every candidate perturbs
every tensor with RMS σ.

The only static degrees of freedom are:

- the stream direction ε;
- the replication pattern, which is shared by all candidates.

Reviewers should be told this directly. "Which layers were hit" is not a
meaningful question for this perturbation family.

**S1b. Verify the stream structure on the real tensor list.**

- **Hypothesis:** for all 642 tensors, the noise equals a reshape of a prefix
  of `randn(622,329,856, bf16, cuda, gen(seed))`.
- **Experiment:** on GPU, compare the regenerated per-tensor noise with the
  slices, for 9504111 and 2 controls.
- **Measurement:** bitwise equality.
- **Falsifier:** any mismatch → the folded-gradient shortcut is invalid. Fall
  back to per-tensor dot products, which are slower but exact.
- **Cost:** about 2 minutes, inside Stage-2 setup.
- **Interpretation:** enables H★ at a cost of seconds per candidate.

**S1c. Effective dimensionality** (CPU arithmetic). The perturbation family
lives in ≤ 622M dimensions rather than 8.77B. Each stream coordinate is
written into as many as ~120 tensors at once, across depth and modality.

## 7. Stage 2: functional localization via first-order geometry (GPU-A)

All functional quantities are defined before inference. They are frozen in a
plan lock committed before the run.

**S2a. Gradient fidelity.**

- **Hypothesis:** a Hugging Face Qwen3-VL forward pass at the pinned revision
  reproduces the stored vLLM base A–D log-probabilities closely enough for
  gradients to be meaningful.
- **Experiment:** HF BF16 forward with the identical prompt and processor on
  all 761 RERANK + TEST examples.
- **Measurement:** max and median |ΔlogP| against the stored vLLM scores, and
  argmax agreement.
- **Falsifier:** argmax agreement < 99%, or median |ΔlogP| > 0.05 nats → stop
  the gradient route.
- **Cost:** about 4 min.
- **Interpretation:** gradients come from HF; behaviour stays with vLLM. This
  is allowed only after this check.

**S2b. Exact candidate in HF layout.**

- **Hypothesis:** applying ε_9504111 in vLLM-packed shapes and splitting
  packed qkv and gate_up rows into HF tensors reproduces the candidate.
- **Experiment:** an HF forward of the reconstructed candidate on 761
  examples.
- **Measurement:** answers and scores against the stored candidate outputs.
- **Falsifier:** answer agreement < 99% → layout mapping error → stop.
- **Cost:** about 4 min.

**S2c. Folded gradients and first-order predictions** (core test of H★).

- **Hypothesis:** t_k[j] ≈ σ⟨fold(∇s_j), ε_k⟩ at the observed scale.
- **Experiment:**
  - One backward pass per example (961 = SEARCH + RERANK + TEST) on the base.
    Fold each gradient into stream coordinates F_j (622M, fp32).
  - Dot F_j with regenerated ε_k:
    - RERANK examples × all 543 RERANK-measured candidates;
    - TEST × all 61 TEST-measured candidates;
    - SEARCH × a fixed, pre-declared 1,000-candidate subsample (250 per σ,
      including the top 50).
  - Set-level folded gradients F_SEARCH, F_RERANK and F_TEST, projected onto
    all 5,000 ε_k.
- **Measurements:**
  - **(i)** For the 13 candidates with A–D scores: per-example correlation
    and slope of predicted against observed t.
  - **(ii)** For all candidates with answers: predicted against observed
    answer flips, using base margins plus predicted t.
  - **(iii)** Across candidates: correlation of predicted and observed set
    gains (SEARCH for 5,000, RERANK for 543, TEST for 61).
  - **(iv)** cos(F_RERANK, F_TEST), cos(F_SEARCH, F_RERANK) and
    cos(F_SEARCH, F_TEST).
- **Falsifiers** (pre-declared):
  - pooled r(t̂, t) < 0.3 for the 13 scored candidates;
  - **or** the predicted-flip rate disagrees with observed flips on more than
    half of 9504111's changed answers;
  - **or** across-candidate r(predicted, observed RERANK gain) < 0.3.

  Any of these → H★ fails and the effects are nonlinear (H-nonlin).
- **Supporting result:**
  - r ≥ 0.6 per example;
  - 9504111 sits in the extreme upper tail of ⟨F_RERANK, ε⟩ and the low tail
    of ⟨F_TEST, ε⟩;
  - cos(F_R, F_T) small or negative.
- **Cost:** about 45–60 min, ≈ $3–3.5.
- **Interpretation:** if supported, the mirage has a *quantitative*
  geometric explanation. The selection-optimism share left unidentified
  earlier becomes estimable from the projections without new questions.

**S2d. Group and depth attribution of the first-order effect** (CPU, from
S2c outputs).

- **Hypothesis:** the first-order account reproduces the 14 *measured*
  group-swap results.
- **Measurement:** compare per-group partial dot products ⟨g_j^(G), δ^(G)⟩
  against the measured insertion and removal shifts from session 1.
- **Falsifier:** disagreement (r < 0.5) → the first-order model fails at
  group level even if it works in aggregate.
- **Cost:** $0, since session-1 data exists.
- **Interpretation:** a free, already-measured causal validation of H★ at
  group resolution. It also says *which* parameter types (attention vs MLP,
  vision vs language, depth) carry F_RERANK vs F_TEST misalignment. That is
  the only localization question that makes sense for this family.

Hidden-state, attention-map and representation-similarity diagnostics are
**deferred**. They answer "what changed" without "why it was selected", and
reviewers rightly discount them. They come in only at Stage 3, and only if
S2d localizes the misaligned component to a small parameter type.

## 8. Stage 3: behavioral computation (CPU first; conditional GPU)

**S3a. Which examples drive F_RERANK − F_TEST?**

- **Hypothesis:** the misaligned component comes from identifiable item
  families (subtask, question form, base-margin regime) rather than spread
  evenly.
- **Experiment:** rank examples by their contribution to ⟨F_D, ε_9504111⟩ and
  to cross-set misalignment, then characterize them by subtask, form, gold
  letter, and close versus far margins. These attributes are fixed before
  looking.
- **Falsifier:** contributions spread uniformly → no item-level computation
  to name.
- **Cost:** $0.

**S3b. Matched probe** (only if S3a isolates a family, e.g. a frame of
reference or egocentric phrasing).

- **Experiment:** a small matched probe set built *only* from the official
  train split (no new benchmark), with minimal edits that keep the image and
  change the targeted property.
- **Falsifier:** 9504111 and controls respond identically.
- **Cost:** ≈ $0.5.
- **Interpretation:** this is where a *visual or spatial* claim could
  legitimately arise. Otherwise none is made.

## 9. Stage 4: causal intervention (GPU-B, original vLLM engine)

All edits act on the *stream* ε in vLLM-packed layout. Each is a single
deterministic construction specified in advance; there is no search.

| # | Intervention | Hypothesis tested | Prediction if H★ holds | Falsifier |
|---|---|---|---|---|
| C1 | 9504111 with its component along F̂_RERANK removed: ε′ = ε − ⟨ε, F̂_R⟩F̂_R, renormalized to ‖ε‖ (necessity) | The RERANK gain is carried by the F_R component | RERANK gain → about the control mean; TEST shifts by the predicted ⟨F_T, removed⟩ | ≥ 50% of the RERANK gain remains |
| C2 | A random control plus 9504111's F̂_R component (transplant, same norm) (sufficiency) | The F_R component alone creates a RERANK "expert" | RERANK gain ≈ predicted; TEST ≈ predicted (a mirage) | No RERANK gain |
| C3 | A pure constructed direction ε = √n · F̂_R at σ (a "designed mirage") | Selection-free construction of the phenomenon | A large RERANK gain and the predicted TEST from cos(F_R, F_T) | RERANK gain ≤ 9504111's |
| C4 | Same as C3 but along F̂_TEST (control for the method) | The method is not RERANK-specific | A TEST gain, RERANK as predicted | No TEST gain |
| C5 | 9504111 with depth coherence broken: each layer's slice sign-flipped by a fixed ±1 pattern, same ‖δ‖ (H-coh) | Depth coherence carries the effect | Effects shrink roughly to the incoherent scale, as predicted from S2d per-layer terms | Effects unchanged |
| C6, C7 | Two second-order checks: −ε (sign-flipped 9504111) and a control (H-flat) | First-order terms flip sign; curvature terms do not | Δ(−ε) ≈ −first-order + same second-order | Symmetric responses |

**Evaluation:**

- RERANK200 + TEST561, A–D scores, generated answers;
- fingerprints recorded;
- exact restore between conditions.

**Cost:** ≈ 7 × 3.6 min + setup ≈ 35 min ≈ **$2.1**.

**Interpretation:**

- C1 + C2 + C3 together would be a *causal* demonstration that RandOpt's
  "expert" is a projection onto the selection set's gradient.
- C3 would construct the mirage without any random search.

C3 and C4 are synthetic directions. They are diagnostic constructions, never
reported as experts.

## 10. Stage 5: replication and generalization

**R1. Within-population, out of sample** (GPU, bundled with B).

- **Experiment:** the 6 pre-declared high-RERANK audit candidates (RERANK
  ≥ +5) evaluated on TEST.
- **Hypothesis:** TEST predicted by ⟨F_T, ε⟩ *before* running.
- **Falsifier:** prediction error larger than the residual SD from S2c.
- **Cost:** ≈ $1.7 (≈ $1.2 if bundled).

**R2. Population-level predictions** ($0, already in S2c). Does
⟨F_SEARCH, ε⟩ predict the SEARCH ranking of all 5,000? Does it predict the σ
dependence (first-order variance ∝ σ², flattening ∝ σ²) and the observed
"mean falls, tail widens" pattern from the radius audit?

**R3. Second task or model** (required for a strong conference claim).
Repeat S2c on one existing population already in the repository:

- the OmniSpatial Complex-Logic population; or
- the Qwen2.5-VL-7B line-tracing population.

That means one model load, set gradients, and projections onto their stored
candidates. Check whether H★'s predictions and the cos(F_selection,
F_heldout) → transfer relation hold there.

- **Cost:** ≈ $3–4 per population, *after* checking that their seeds,
  outputs and RNG recipe are recoverable (CPU, $0).
- **Interpretation:** turns a single-candidate case study into a general
  account of random-weight search.

## 11. What would constitute a genuine mechanistic result

All of the following, on frozen definitions:

1. **Validity.** S2c holds:
   - per-example r ≥ 0.6;
   - candidate-level r(predicted, observed RERANK) ≥ 0.5;
   - S2d reproduces the measured group swaps.
2. **Explanation of the mirage.**
   - 9504111 lies in the upper tail of ⟨F_R, ε⟩;
   - cos(F_R, F_T) is low or negative;
   - the predicted TEST matches the observed −2.67 within the residual SD.
3. **Necessity and sufficiency.**
   - C1 removes most of the RERANK gain (necessity);
   - C2 or C3 creates a RERANK "expert" with the predicted TEST failure
     (sufficiency).
4. **Out-of-sample prediction.** R1 predicts held-out TEST outcomes of
   unseen high-RERANK candidates.
5. **Replication.** R3 shows the same relation on a second population.

Items 1–3 give a strong single-system result. Items 4–5 make it general.

## 12. What would constitute a NO-GO

| Condition | Meaning |
|---|---|
| S2a/S2b fidelity fails | The gradient route is unavailable. Keep the empirical paper. |
| S2c falsifiers trigger | The effects are not first-order; the behaviour is genuinely nonlinear. A compact mechanism is then unlikely, given distributed group effects. Report a NO-GO for mechanism. |
| S2c holds but C1–C3 fail | Correlational only. No causal claim. |
| R3 fails | A curiosity of one population; workshop or short-paper scope. |

The circuit route (H-vis / H-head) is already disfavoured by measured
evidence. It is **not** pursued unless S2d localizes the misaligned component
to a small, specific parameter type.

## 13. Minimum GPU budget (measured rate $3.49/h)

| Session | Contents | Estimate | Hard cap |
|---|---|---:|---:|
| GPU-A | Setup; S1b; S2a; S2b; S2c (961 backward passes, folded dots for 543 / 61 / 1,000 candidates, set projections for 5,000) | ≈ 60 min ≈ **$3.5** | $5.00 |
| GPU-B | C1–C7 (7 conditions × 761) + R1 (6 TEST runs), one session | ≈ 60 min ≈ **$3.5** | $5.00 |
| GPU-C (optional, required for the strong claim) | R3 on one existing population | ≈ $3–4 | $5.00 |
| **Minimum to reach a GO / NO-GO on mechanism** | GPU-A only | **$3.5** | **$5.00** |

GPU-B runs only if GPU-A passes its falsifiers. GPU-C runs only if GPU-B
passes.

## 14. Recommended implementation order

1. **CPU, now:**
   - Stage 0 spec;
   - write and lock the GPU-A plan (definitions, falsifiers, candidate lists,
     the SEARCH subsample);
   - implement HF folding, packed↔HF mapping and stream regeneration, with
     unit tests on a toy model;
   - check that the populations needed for R3 are recoverable.
2. **GPU-A** (≈ $3.5, cap $5): S1b → S2a → S2b → S2c. Stop at the first failed
   fidelity check.
3. **CPU:** S2d and S3a. Decide GO or NO-GO against the pre-declared
   falsifiers.
4. **GPU-B** (only on GO): C1–C7 + R1.
5. **CPU:** the causal analysis, then a final GO or NO-GO on the mechanistic
   claim.
6. **GPU-C** (only on GO): R3 replication.
7. **Then** write the paper.

## 15. Potential strong-conference paper claim (only if items 1–5 of §11 hold)

> *Random weight-space search selects gradient projections, not experts.*
>
> - In a VLM, per-tensor-reseeded perturbations (the RandOpt family) live in a
>   ≤ 622M-dimensional, depth-replicated subspace.
> - Their effects are, to first order, projections onto each evaluation set's
>   folded gradient.
> - Selection therefore performs an implicit, noisy gradient step on the
>   selection set.
> - "Experts" fail to transfer exactly to the degree that selection-set and
>   test-set folded gradients are misaligned, which we predict before
>   evaluation.
> - Removing the selected gradient component erases the expert; transplanting
>   it creates one; a constructed gradient direction reproduces the mirage
>   with no search at all.
> - The relation replicates on a second population.

## Blunt assessment

**The visual-circuit story you hoped for is unlikely.** The perturbation is a
single Gaussian stream replicated across all layers and modalities. Its
measured effects are distributed and roughly additive, with no group above
36%. A "localized visual computation" claim would very probably be an
elaborate story built around an accident. I would not pursue it unless S2d
localizes the effect, and I expect it will not.

**The strongest defensible path is the gradient-geometry account (H★).** It
is cheap to falsify (≈ $3.5 for GPU-A) and makes quantitative, pre-declarable
predictions across thousands of candidates. It can be tested causally with
constructed perturbations, which is unusually clean.

**Executed successfully, it plausibly reaches a strong conference, but only
with the R3 replication and the C1–C3 causal results.** Even then, expect a
reviewer to say "random search ≈ zeroth-order gradient estimation is known."
The novelty must come from three things:

- the false-expert phenomenon, explained and *predicted* quantitatively from
  cross-set gradient alignment;
- the constructed mirage;
- the structural finding that RandOpt's reseeded perturbations collapse to a
  depth-replicated subspace. That finding bears directly on the "Neural
  Thickets" density claims.

**Without R3 it is a strong single-system analysis**, borderline main-track.
**If S2c fails** (non-first-order effects), there is no compact mechanism to
find, and the honest outcome is the existing empirical paper plus a NO-GO on
mechanism.
