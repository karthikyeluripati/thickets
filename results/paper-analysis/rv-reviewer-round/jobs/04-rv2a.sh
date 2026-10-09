#!/usr/bin/env bash
# RV-2a: RandOpt's search under the boxed prompt, Qwen2.5-3B-Instruct, population seed 42 (= C3B's perturbations).
set -ux
grep -q '^rc=0 ' /workspace/jobs/01-prep.rc
. /workspace/repo/scripts/pod_rv_common.sh
CAP=$(grep -o 'CAP=[0-9.]*' /workspace/rate.txt | cut -d= -f2)
search_row q3 /workspace/rv/q3 "$(cat /workspace/mp_q3.txt)" boxed 42 84.0 90.0 82.34 $A/c3b-sameRun/pod/c3b/out/randopt.log $CAP
