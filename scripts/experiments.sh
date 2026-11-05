#!/usr/bin/env bash
# Task 4 / Task 7 experiments. Each writes its evidence to results/task4/<name>/ (or results/ for the
# standard run). Usage: scripts/experiments.sh {hpa|rollout|recovery|coldstart|lb|standard}
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-http://localhost:8080}
LOCUST=.venv/bin/locust
NS=app

ts() { date -u +%H:%M:%S; }

# background sampler: replica / readiness / HPA / endpoint state every 2 s
watch_state() {
  local out=$1
  while true; do
    {
      echo "--- $(ts)"
      kubectl -n $NS get deploy parser --no-headers -o custom-columns='DESIRED:.spec.replicas,READY:.status.readyReplicas,UPDATED:.status.updatedReplicas,AVAILABLE:.status.availableReplicas' 2>/dev/null | sed 's/^/deploy desired,ready,updated,available: /'
      kubectl -n $NS get hpa parser --no-headers 2>/dev/null | sed 's/^/hpa: /'
      kubectl -n $NS get pods -l app=parser --no-headers -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,READY:.status.containerStatuses[0].ready,VERSION:.spec.containers[0].env[1].value' 2>/dev/null | sed 's/^/pod: /'
      kubectl -n $NS get endpoints parser -o jsonpath='{"endpoints ready="}{.subsets[*].addresses[*].targetRef.name}{" notReady="}{.subsets[*].notReadyAddresses[*].targetRef.name}{"\n"}' 2>/dev/null
    } >> "$out"
    sleep 2
  done
}

top_pods() {
  local out=$1
  while true; do
    { echo "--- $(ts)"; kubectl -n $NS top pods -l app=parser --no-headers 2>/dev/null; } >> "$out"
    sleep 10
  done
}

locust_run() {  # locust_run OUTDIR RUNTIME
  local d=$1 rt=$2
  LOADTEST_HEADERS_OUT="$d/loadtest_headers.json" $LOCUST -f tools/locustfile.py --headless --users 4 \
    --spawn-rate 4 --run-time "$rt" --host "$HOST" --csv "$d/loadtest" > "$d/locust_output.txt" 2>&1
  echo "locust exit code $?" >> "$d/locust_output.txt"
}

summary() {  # print requests / failures / failure% / p50 / p95 / rps from the Aggregated row
  python3 - "$1/loadtest_stats.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
a = [r for r in rows if r["Name"] == "Aggregated"][0]
n, f = int(a["Request Count"]), int(a["Failure Count"])
print(f"requests={n} failures={f} failure_rate={100*f/max(n,1):.3f}% p50={a['50%']}ms p95={a['95%']}ms rps={float(a['Requests/s']):.2f}")
EOF
}

pin_two() {   # raise the HPA floor to 2 for experiments that must start from two Ready replicas
  kubectl -n $NS patch hpa parser -p '{"spec":{"minReplicas":2}}' >/dev/null && echo "$(ts) HPA minReplicas temporarily 2"
}
unpin() {
  kubectl -n $NS patch hpa parser -p '{"spec":{"minReplicas":1}}' >/dev/null && echo "$(ts) HPA minReplicas restored to 1"
}

wait_two_ready() {
  for _ in $(seq 1 120); do
    [ "$(kubectl -n $NS get deploy parser -o jsonpath='{.status.readyReplicas}')" = "2" ] && return 0; sleep 2
  done
  echo "timed out waiting for 2 ready replicas" >&2; return 1
}

case "${1:-}" in
  hpa)
    d=results/task4/hpa; rm -rf "$d"; mkdir -p "$d"
    kubectl -n $NS get hpa parser -o yaml > "$d/hpa.yaml"
    watch_state "$d/state.txt" & W=$!
    top_pods "$d/top.txt" & T=$!
    echo "$(ts) waiting for the HPA to settle at 1 replica (idle)" | tee "$d/timeline.txt"
    for _ in $(seq 1 120); do [ "$(kubectl -n $NS get deploy parser -o jsonpath='{.status.readyReplicas}')" = "1" ] && break; sleep 5; done
    echo "$(ts) at 1 replica; starting 4-minute load" | tee -a "$d/timeline.txt"
    locust_run "$d" 4m
    echo "$(ts) load finished; waiting for scale-down" | tee -a "$d/timeline.txt"
    for _ in $(seq 1 60); do [ "$(kubectl -n $NS get deploy parser -o jsonpath='{.status.readyReplicas}')" = "1" ] && break; sleep 5; done
    echo "$(ts) back at $(kubectl -n $NS get deploy parser -o jsonpath='{.status.readyReplicas}') replica(s)" | tee -a "$d/timeline.txt"
    kill $W $T
    kubectl -n $NS describe hpa parser > "$d/hpa_describe.txt"
    kubectl -n $NS get events --sort-by=.lastTimestamp | grep -iE "horizontal|scaled" > "$d/scaling_events.txt"
    summary "$d" | tee -a "$d/timeline.txt"
    ;;

  rollout)
    d=results/task4/rollout; rm -rf "$d"; mkdir -p "$d"
    pin_two | tee -a "$d/timeline.txt"; trap unpin EXIT
    wait_two_ready
    watch_state "$d/state.txt" & W=$!
    echo "$(ts) start 5-minute load" | tee "$d/timeline.txt"
    locust_run "$d" 5m & L=$!
    sleep 40
    echo "$(ts) kubectl set env: MODEL_VERSION=v2 MODEL_URI=s3://models/v2/" | tee -a "$d/timeline.txt"
    kubectl -n $NS set env deploy/parser MODEL_VERSION=v2 MODEL_URI=s3://models/v2/ | tee -a "$d/timeline.txt"
    kubectl -n $NS rollout status deploy/parser --timeout=300s | tee -a "$d/timeline.txt"
    echo "$(ts) rolled out v2: $(curl -s -D - -o /dev/null -XPOST $HOST/parse -H 'Content-Type: application/json' -d '{"text":"what time is it in tokyo"}' | grep -i x-model-version | tr -d '\r')" | tee -a "$d/timeline.txt"
    sleep 40
    echo "$(ts) kubectl rollout undo" | tee -a "$d/timeline.txt"
    kubectl -n $NS rollout undo deploy/parser | tee -a "$d/timeline.txt"
    kubectl -n $NS rollout status deploy/parser --timeout=300s | tee -a "$d/timeline.txt"
    wait $L
    kill $W
    kubectl -n $NS rollout history deploy/parser > "$d/rollout_history.txt"
    for i in 1 2 3 4 5 6; do
      curl -s -D - -o /dev/null -XPOST $HOST/parse -H 'Content-Type: application/json' -d '{"text":"what time is it in tokyo"}' | grep -iE "x-model-version|x-served-by" | tr -d '\r' | paste -sd' ' -
    done | tee "$d/after_rollback_headers.txt"
    summary "$d" | tee -a "$d/timeline.txt"
    cat "$d/loadtest_headers.json"
    ;;

  recovery)
    d=results/task4/recovery; rm -rf "$d"; mkdir -p "$d"
    pin_two | tee -a "$d/timeline.txt"; trap unpin EXIT
    wait_two_ready
    watch_state "$d/state.txt" & W=$!
    echo "$(ts) start 3-minute load" | tee "$d/timeline.txt"
    locust_run "$d" 3m & L=$!
    sleep 40
    victim=$(kubectl -n $NS get pods -l app=parser -o jsonpath='{.items[0].metadata.name}')
    echo "$(ts) deleting pod $victim" | tee -a "$d/timeline.txt"
    kubectl -n $NS get pods -l app=parser -o wide | tee -a "$d/timeline.txt"
    kubectl -n $NS delete pod "$victim" --wait=false | tee -a "$d/timeline.txt"
    wait_two_ready && echo "$(ts) replica count restored: $(kubectl -n $NS get deploy parser -o jsonpath='{.status.readyReplicas}')/2 ready" | tee -a "$d/timeline.txt"
    kubectl -n $NS get pods -l app=parser -o wide | tee -a "$d/timeline.txt"
    wait $L
    kill $W
    summary "$d" | tee -a "$d/timeline.txt"
    ;;

  coldstart)
    d=results/task4/coldstart; rm -rf "$d"; mkdir -p "$d"
    pin_two | tee -a "$d/timeline.txt"; trap unpin EXIT
    for run in 1 2 3; do
      wait_two_ready
      watch_state "$d/state_run$run.txt" & W=$!
      old=$(kubectl -n $NS get pods -l app=parser -o jsonpath='{.items[*].metadata.name}')
      echo "$(ts) run $run: deleting all pods: $old" | tee -a "$d/timeline.txt"
      kubectl -n $NS delete pods -l app=parser --wait=false >/dev/null
      sleep 3
      wait_two_ready
      kill $W
      for p in $(kubectl -n $NS get pods -l app=parser -o jsonpath='{.items[*].metadata.name}'); do
        echo "$old" | grep -qw "$p" && continue
        created=$(kubectl -n $NS get pod "$p" -o jsonpath='{.metadata.creationTimestamp}')
        ready=$(kubectl -n $NS get pod "$p" -o jsonpath='{.status.conditions[?(@.type=="Ready")].lastTransitionTime}')
        secs=$(python3 -c "from datetime import datetime as D; f='%Y-%m-%dT%H:%M:%SZ'; print(int((D.strptime('$ready',f)-D.strptime('$created',f)).total_seconds()))")
        echo "run=$run pod=$p created=$created ready=$ready seconds=$secs" | tee -a "$d/cold_start_times.txt"
        kubectl -n $NS logs "$p" > "$d/log_run${run}_$p.txt"
      done
    done
    python3 - "$d/cold_start_times.txt" <<'EOF' | tee -a "$d/cold_start_times.txt"
import re, statistics, sys
runs = {}
for line in open(sys.argv[1]):
    m = re.search(r"run=(\d+) .* seconds=(\d+)", line)
    if m:
        runs.setdefault(int(m.group(1)), []).append(int(m.group(2)))
per_run = {r: max(v) for r, v in sorted(runs.items())}   # a run is done when its slowest pod is Ready
print(f"per-run cold start (slowest pod, s): {per_run}")
print(f"median over 3 runs: {statistics.median(per_run.values())} s")
allv = [x for v in runs.values() for x in v]
print(f"all pods: {sorted(allv)}; median per pod: {statistics.median(allv)} s")
EOF
    grep -h "model download\|downloaded s3://models\|ready in" "$d"/log_run1_* | head -12 > "$d/download_evidence.txt"
    ;;

  lb)
    d=results/task4/loadbalance; rm -rf "$d"; mkdir -p "$d"
    # The test needs exactly two Ready replicas for the whole run, so the HPA floor is raised to 2
    # for its duration and restored afterwards (max stays 2).
    kubectl -n $NS patch hpa parser -p '{"spec":{"minReplicas":2}}' | tee "$d/timeline.txt"
    wait_two_ready
    kubectl -n $NS get pods -l app=parser -o wide | tee -a "$d/timeline.txt"
    watch_state "$d/state.txt" & W=$!
    top_pods "$d/top.txt" & T=$!
    locust_run "$d" 2m
    kill $W $T
    kubectl -n $NS patch hpa parser -p '{"spec":{"minReplicas":1}}' | tee -a "$d/timeline.txt"
    summary "$d" | tee -a "$d/timeline.txt"
    cat "$d/loadtest_headers.json"
    ;;

  standard)
    # Task 7: the exact marking command, writing to results/loadtest_*
    kubectl -n $NS patch hpa parser -p '{"spec":{"minReplicas":2}}'
    wait_two_ready
    rm -f results/kubectl_top_during_loadtest.txt
    top_pods results/kubectl_top_during_loadtest.txt & T=$!
    $LOCUST -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host "$HOST" \
      --csv results/loadtest 2>&1 | tee results/loadtest_output.txt
    kill $T
    kubectl -n $NS patch hpa parser -p '{"spec":{"minReplicas":1}}'
    ;;

  *) echo "usage: $0 {hpa|rollout|recovery|coldstart|lb|standard}"; exit 1 ;;
esac
