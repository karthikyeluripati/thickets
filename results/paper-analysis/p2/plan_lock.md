# P2 lock: is RandOpt's reported advantage over majority vote robust to a properly-sampled self-consistency baseline?

Locked before any P2 output. User-approved pod, cap **$10**. Context: `../p1/P1_RESULT.md`.
- The Neural Thickets paper (arXiv 2603.12228v1) reports RandOpt (K = 50) ahead of "TT-MV", defined as "majority vote
  over test samples from a single trained model with different seeds", with the **temperature unstated**. RandOpt's
  own code generates at temperature 0.
- On GQA the paper reports no majority-vote baseline.

## Comparison targets (paper's numbers, their runs)
| Task / model | paper Base | paper TT-MV | paper RandOpt (K = 50) |
|---|---|---|---|
| GSM8K / Qwen2.5-0.5B-Instruct | 39.9 | 41.0 | 54.1 |
| GSM8K / Qwen2.5-1.5B-Instruct | 58.8 | 69.1 | 76.4 |
| GQA testdev / Qwen2.5-VL-3B-Instruct | 56.6 | – | 69.0 |

## Our protocol
- RandOpt's prompts and RandOpt's own scorers:
  - **GSM8K:** the verl preprocessing prompt (question + ' Let's think step by step and output the final answer after
    "####".'), chat template, max_tokens 1024, GSM8KHandler (strict then flexible).
  - **GQA:** the CoT + \boxed prompt, max_tokens 256, GQAHandler.
- **Data:** GSM8K is the full test set (1319). GQA is 2000 items by hash from testdev-balanced (the paper uses all
  12,578).
- **Conditions:**
  - greedy (n = 1);
  - SC at T = 0.7 (primary) and T = 0.3 (reported), n = 50, top_p 1.0, fixed seed.
- **Vote:** majority over extracted answers (handler.extract_answer, exact string). The voted answer is correct iff the
  scorer accepts it (GSM8K: equals the gold number; GQA: GQAHandler match). SC@K curves use the first K samples,
  K ∈ {1, 5, 10, 20, 50}.
- **Models:** Qwen2.5-{0.5B, 1.5B}-Instruct (Hugging Face main, revision recorded); Qwen2.5-VL-3B @ 66285546.

## Gate
Our greedy base must reproduce the paper's Base within ±2.0 pp (GQA: ±2.5 pp, since we use a subset). Otherwise that
row is **NOT COMPARABLE**.

## Decision per row (SC@50, T = 0.7, vs the paper's RandOpt)
**MATCHES** if SC ≥ RandOpt − 1.0; **RANDOPT AHEAD** if SC ≤ RandOpt − 3.0; else **CLOSE**.
- Item-bootstrap 95% CIs of our SC accuracy are reported.
- Also reported: SC@50 at T = 0.7 minus the paper's TT-MV.

## Caveats (stated in advance)
- These are cross-paper comparisons: different hardware and software, and the paper averages 3 runs. The base gate
  controls the gross setup.
- RandOpt's training-time compute (5000 × 200 evaluations + K = 50 models) is not matched. SC uses 50 samples of one
  model at test time.
- **A MATCH would mean** RandOpt's reported gains are reproducible without weight search. **RANDOPT AHEAD would
  support** their claim over properly-sampled SC.
