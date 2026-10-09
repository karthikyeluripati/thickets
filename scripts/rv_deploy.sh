#!/usr/bin/env bash
# Local: stage the RV session on a fresh pod and start the job queue (results/paper-analysis/rv-reviewer-round/plan_lock.md).
# usage: scripts/rv_deploy.sh <host> <port> <cap-usd>      (key ~/.ssh/id_ed25519)
set -eu
H=$1; P=$2; CAP=$3; S="ssh -p $P -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@$H"
A=results/paper-analysis
$S "mkdir -p /workspace/repo /workspace/jobs"
tar cz scripts $A/rv-reviewer-round/jobs $A/g2-sameRun/pod/g2/items.json $A/g2-sameRun/pod/g2/out/topk.json $A/g1/frozen_items.json \
  $A/o2-olmo-prompt/top50_reconstructed.json $A/c3b-sameRun/pod/c3b/out/randopt.log $A/c-sameRun/pod6/c/out/randopt.log \
  | $S "tar xz -C /workspace/repo && cp /workspace/repo/$A/rv-reviewer-round/jobs/*.sh /workspace/jobs/ && ls /workspace/jobs"
$S "cd /workspace && nohup bash /workspace/repo/scripts/pod_jobqueue.sh $CAP Qwen/Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1 > /workspace/queue.log 2>&1 & sleep 2; cat /workspace/rate.txt"
echo DEPLOYED
