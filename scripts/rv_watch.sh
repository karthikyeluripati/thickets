#!/usr/bin/env bash
# Local watcher: every 3 minutes copy the RV outputs off the pod into results/paper-analysis/rv-reviewer-round/pod/;
# when /workspace/DONE appears, take a final copy and touch /workspace/RETRIEVED (the pod guard then stops the pod).
# usage: scripts/rv_watch.sh <host> <port>
set -u
H=$1; P=$2; S="ssh -p $P -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@$H"
D=results/paper-analysis/rv-reviewer-round/pod; mkdir -p $D
pull() { $S "cd /workspace && tar cz --exclude='*.texts.json.gz' --exclude='smoke_*' rv jobs/*.log jobs/*.rc rate.txt guard.log gpu.txt queue.log 2>/dev/null" | tar xz -C $D 2>/dev/null
         $S "cd /workspace && tar cz rv/*/out/*.texts.json.gz rv/rv1 rv/rv5 2>/dev/null" | tar xz -C $D 2>/dev/null; date -u +%FT%TZ; }
while true; do
  pull
  if $S "test -e /workspace/DONE" 2>/dev/null; then pull; $S "touch /workspace/RETRIEVED"; echo RETRIEVED; break; fi
  $S "true" 2>/dev/null || { echo "POD_UNREACHABLE $(date -u +%FT%TZ)"; }
  sleep 180
done
