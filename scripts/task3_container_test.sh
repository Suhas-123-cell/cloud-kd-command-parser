#!/usr/bin/env bash
# Task 3: run the service container alone with --cpus 1 --memory 1g, send 50 sequential
# /parse requests, and record latency, status codes, OOM state and peak memory.
# Needs SeaweedFS reachable at localhost:8333 (k3d port mapping) and .secrets/s3.env.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/task3}
IMG=${IMG:-cc-kd/parser:1.0}
NAME=parser-limits
mkdir -p "$OUT"
set -a; source .secrets/s3.env; set +a

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p 127.0.0.1:8000:8000 --cpus 1 --memory 1g \
  -e MODEL_URI=s3://models/v1/ -e MODEL_VERSION=v1 -e MAX_NEW_TOKENS=64 \
  -e S3_ENDPOINT=http://host.docker.internal:8333 \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY "$IMG" >/dev/null

# sample docker stats once a second in the background
( while docker inspect "$NAME" >/dev/null 2>&1; do
    docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' "$NAME" 2>/dev/null | sed "s/^/$(date +%T) /"
    sleep 1
  done ) > "$OUT/docker_stats.txt" &
STATS_PID=$!

echo "waiting for /readyz ..."
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/readyz)" = 200 ] && break; sleep 1
done
curl -s -i localhost:8000/readyz | tee "$OUT/readyz.txt"; echo

python3 - "$OUT" <<'EOF'
import json, sys, time, urllib.request, statistics
out = sys.argv[1]
texts = [json.loads(l)["text"] for l in open("data/validation.jsonl")][:50]
rows, lat = [], []
for i, t in enumerate(texts, 1):
    req = urllib.request.Request("http://localhost:8000/parse", data=json.dumps({"text": t}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            code, body, hdr = r.status, r.read().decode(), dict(r.headers)
    except urllib.error.HTTPError as e:
        code, body, hdr = e.code, e.read().decode(), dict(e.headers)
    ms = (time.time() - t0) * 1000
    lat.append(ms)
    rows.append({"n": i, "status": code, "ms": round(ms, 1), "x_model_version": hdr.get("x-model-version"),
                 "x_served_by": hdr.get("x-served-by"), "text": t, "body": body})
    print(f"{i:2d} {code} {ms:7.1f} ms  {body}")
with open(f"{out}/requests.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
s = sorted(lat)
summary = {"requests": len(rows), "status_counts": {str(c): sum(r["status"] == c for r in rows) for c in {r["status"] for r in rows}},
           "latency_ms_mean": round(statistics.mean(lat), 1), "latency_ms_p50": round(s[len(s) // 2], 1),
           "latency_ms_p95": round(s[int(len(s) * 0.95) - 1], 1), "latency_ms_max": round(s[-1], 1)}
print(json.dumps(summary, indent=2))
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
EOF

{
  echo "# cgroup v2 counters read inside the container after the 50 requests"
  echo "memory.max  = $(docker exec "$NAME" cat /sys/fs/cgroup/memory.max)"
  echo "memory.peak = $(docker exec "$NAME" cat /sys/fs/cgroup/memory.peak) bytes"
  echo "cpu.max     = $(docker exec "$NAME" cat /sys/fs/cgroup/cpu.max)"
  echo "memory.events:"; docker exec "$NAME" cat /sys/fs/cgroup/memory.events
  echo; echo "# docker inspect"
  docker inspect "$NAME" --format 'OOMKilled={{.State.OOMKilled}} Status={{.State.Status}} RestartCount={{.RestartCount}} NanoCpus={{.HostConfig.NanoCpus}} Memory={{.HostConfig.Memory}}'
} | tee "$OUT/memory_peak.txt"

kill "$STATS_PID" 2>/dev/null || true
docker logs "$NAME" > "$OUT/container_log.txt" 2>&1
docker rm -f "$NAME" >/dev/null
echo "peak from docker stats samples:"; sort -t= -k3 "$OUT/docker_stats.txt" | awk '{print $4}' | sort -h | tail -1
