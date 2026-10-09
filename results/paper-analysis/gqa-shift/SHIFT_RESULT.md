# EXPLORATORY: what RandOpt's selected GQA models do differently (not pre-registered)

Script `scripts/gqa_shift_analysis.py` on the G3 outputs (G2's top-50 members, BASE, SC@50; 1238 questions).
Outputs `shift_b256.json` (primary, the paper's budget) and `shift_b1024.json` (same conclusions). CPU only.
Sanity check: the analysis's approximate vote scoring gives D = +3.63 at 256, equal to the locked G3 value.

## 1. Selected models mostly stop reasoning and answer directly
| 256 tokens | base | selected members | SC samples |
|---|---|---|---|
| mean / median tokens | 147 / 146 | 66 / **10** | 143 / 143 |
| direct answer (≤ 32 tokens) | 3.0% | **55.8%** | 5.9% |
| step-by-step reasoning in text | 97.0% | **43.7%** | – |
| boxed final answer | 88.3% | 90.2% | 83.9% |
| hit max tokens | 1.9% | 1.2% | 2.7% |

Members vary from 13 to 157 mean tokens (direct share 1%–96%; median member 66%). **Across the 50 members, accuracy and
mean length correlate r = −0.89** (−0.87 at 1024): the shorter a member answers, the more accurate it is.
Within a question, members' shorter generations are correct **+7.1 pp [5.1, 9.2]** more often than their longer ones
(780 questions with a split; +6.4 [4.4, 8.5] at 1024).

## 2. The gain is content on open questions, not termination
Member gain over base +5.0 pp, all on open questions (+7.7 pp; yes/no +0.1). By base outcome (share of the net gain):
questions base answers properly but wrongly (n = 444) **+200%** (members 27.9% correct); questions base fails to
finish (n = 133) +90% (members 41.9%); questions base gets right (n = 661) **−190%** (members 82.2%). Selection fixes
far more properly-answered errors than unfinished ones, at a real cost on what base already gets right.

## 3. Selection re-weights answers the base model can already give
On the 92 questions only the RandOpt vote gets right (SC wrong): the correct answer is among SC's 50 base samples in
**95.7%** of them, but at a median share of 14% (SC's winning wrong answer: 28%). Among the members it has a median
share of 51%. Base greedy is correct on 25% of these. Selection does not create new answers; it moves probability
onto answers the base model gives rarely when it reasons step by step.

## 4. Answer form on open questions
Answer length (words): ground truth 1.08, base 1.58, members 1.31, SC samples 1.52; exact key = ground truth: base
28.3%, members 35.8%; answers longer than 3 words: base 5.0%, members 1.9%.

## Reading and open question
On GQA with this 3B vision-language model, step-by-step reasoning hurts. RandOpt's selected perturbations mostly
switch it off (direct, short answers), which re-weights the base model's own answers toward the right ones on open
questions. That is a shared, transferable shift, so voting over selected models beats voting over base samples (which all
reason). **This analysis cannot say whether weight search is needed for it:** if prompting the base model to answer
directly (no chain of thought) gives the same gain, RandOpt's GQA advantage is a prompt effect it finds by search. That
needs a GPU control (base greedy and SC@50 with a direct-answer prompt, same questions).

> **Resolved later:** G4 (`g4-direct-prompt/`) ran that control (one direct-prompt generation 64.70 vs RandOpt 63.49;
> RandOpt − SC@50 direct −1.21 [−2.83, +0.40]), and GD (`gd-gqa-direct-search/`) re-ran RandOpt's whole search under
> the direct prompt (equivalent to SC: −0.32 [−1.29, +0.65]).
