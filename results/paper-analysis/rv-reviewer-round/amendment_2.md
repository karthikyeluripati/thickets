# RV amendment 2: RV-2b is run after all (session 4)

Written before any RV-2b output. The user offered a further GPU budget after amendment 1 had set RV-2b and RV-3 to
"not run". RV-2b (Qwen2.5-1.5B, RandOpt's search under the boxed prompt) is run exactly as locked in `plan_lock.md`
(design, gates — ENV against PS's 68.5 / 80.5 and Q2's boxed BASE 70.20, fidelity against C's randopt.py log — budget
rule, primary and secondary; analysis `gb_analysis.py` with C's ensemble and top-k seeds). One 4× H100 pod, guard cap
$19 (the pods of this project have been stopped near $20). Prep now retries and verifies the model download (session 2's
prep recorded an empty path after a failed download). **RV-3 (second OLMo seed) remains not run** (budget; reported).
