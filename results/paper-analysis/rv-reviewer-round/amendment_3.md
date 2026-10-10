# RV amendment 3: RV-3 is run after all (session 5)

Written before any RV-3 output. The user offered a final GPU budget (≈ $17.67) after amendment 2. RV-3 (a second
OLMo-2-1B search under RandOpt's own prompt, population seed 43, fast runner) is run exactly as locked in `plan_lock.md`:
ENV gate (this runner's BASE selection accuracy under RandOpt's prompt within 1.0 pp of 41.5; `o2_eval.py` BASE test
within 1.0 pp of O2's 33.66), budget rule (smoke projection ≤ cap, else not run), primary D = RandOpt (seed 43) −
SC@50 (RandOpt's prompt, O1's samples), secondary as locked; analysis `rv_analysis.py` (its RV-3 block). One 4× H100
pod, guard cap $17. Prep downloads and verifies the model as in amendment 2. If the budget rule stops RV-3, it is
reported as not run.
