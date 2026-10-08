# G2R amendment 1 (budget only; before any full-run output; no change to items, population, rules or analysis)

The smoke projection was $29.00 > the $28 gate (`pod/g2r/projection.txt`), so the full run did not start, as locked.
G2's full run measured 9.65 s per perturbation per worker (mean over 5000; smoke had shown 7.25), so the realistic total
is about $30. The user chose to proceed: session cap raised **$30 → $36** (pod guard replaced, same logic), projection
gate waived. Smoke engineering event: second-wave workers 2 and 3 failed vLLM's memory-profiling check because workers 0
and 1 finished their 2 smoke perturbations and released memory during it; both were re-run alone (resumable), as locked.
