# keep the pod (and its results) alive up to 75 min after G3 so results can be pulled once local disk space is freed
for i in $(seq 1 450); do [ -e /workspace/RETRIEVED ] && exit 0; sleep 10; done
echo WAIT_TIMEOUT
