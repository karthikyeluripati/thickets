#!/usr/bin/env bash
# RV-3: a second RandOpt search on OLMo-2-1B under RandOpt's own prompt, population seed 43 (as G2R), fast runner.
set -ux
grep -q '^rc=0 ' /workspace/jobs/01-prep.rc
. /workspace/repo/scripts/pod_rv_common.sh
CAP=$(grep -o 'CAP=[0-9.]*' /workspace/rate.txt | cut -d= -f2)
search_row olmo2 /workspace/rv/olmo2 "$(cat /workspace/mp_olmo.txt)" randopt 43 41.5 41.5 33.66 none $CAP
