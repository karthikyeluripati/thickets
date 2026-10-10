#!/usr/bin/env bash
# RV-2b: RandOpt's search under the boxed prompt, Qwen2.5-1.5B-Instruct, population seed 42 (= C's perturbations).
set -ux
grep -q '^rc=0 ' /workspace/jobs/01-prep.rc
. /workspace/repo/scripts/pod_rv_common.sh
CAP=$(grep -o 'CAP=[0-9.]*' /workspace/rate.txt | cut -d= -f2)
search_row q15 /workspace/rv/q15 "$(cat /workspace/mp_q15.txt)" boxed 42 68.5 80.5 70.20 $A/c-sameRun/pod6/c/out/randopt.log $CAP
