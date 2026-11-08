# Lightweight command parser: phi3.5 → pythia-70m on a local cloud

CS3081 Cloud Computing mini project · Suhas Chowdary · Roll no. **230101094**

**GitHub repository:** https://github.com/Suhas-123-cell/cloud-kd-command-parser
**Model release (v1, v2, checksums, version info):** https://github.com/Suhas-123-cell/cloud-kd-command-parser/releases/tag/models-r2
**Report:** [`report/report.pdf`](report/report.pdf) · **AI use:** [`AI_USE.md`](AI_USE.md)

A teacher (phi3.5, 3.8 B, via Ollama, temperature 0, seed 42, JSON-schema structured output) labels 3,000 commands in two parallel
Kubernetes workers. The cleaned replies train EleutherAI/pythia-70m (sequence-level knowledge
distillation, CPU only). The student is served by a FastAPI container on a k3d cluster that pulls its
weights from SeaweedFS (S3) at start-up, behind Traefik, with probes, an HPA and rolling updates.

| Model (validation.jsonl, 100 commands) | JSON validity | Intent | Full frame |
|---|---:|---:|---:|
| Teacher phi3.5 | 100 % | 84 % | 55 % |
| Untrained pythia-70m | 0 % | 0 % | 0 % |
| **Student v1 (served)** | **100 %** | **85 %** | **56 %** |
| Student v2 | 99 % | 86 % | 56 % |

Standard load test through the Ingress (2 replicas, 4 users, 2 min): 738 requests, **0 failures**,
p50 630 ms, p95 1,000 ms, 6.17 req/s. Raw evidence for every number is in [`results/`](results/).

## Prerequisites

Docker (Docker Desktop on macOS, or Docker Engine on Linux), [k3d](https://k3d.io) ≥ 5.7, kubectl,
curl, openssl, and either [uv](https://docs.astral.sh/uv/) or Python 3.12 (torch 2.5.1 has no wheels
for Python ≥ 3.13). Free local ports: 8080 (Ingress), 8333 (S3), 6550 (Kubernetes API). About 6 GB
of RAM for Docker. Ollama with `phi3.5` is only needed to re-run the teacher (`make label`).

## Build and deploy (three commands from a fresh clone)

```bash
git clone https://github.com/Suhas-123-cell/cloud-kd-command-parser.git
cd cloud-kd-command-parser
make up
```

`make up` creates the cluster from `cluster/k3d.yaml`, deploys SeaweedFS in namespace `data`, generates
random S3 credentials into the git-ignored `.secrets/` and loads them as Kubernetes Secrets, creates
the `datasets` and `models` buckets and uploads the data, downloads `model-v1.tar.gz` and
`model-v2.tar.gz` from the GitHub Release (verifying each SHA-256 against `models/vN/model.sha256`)
into `s3://models/vN/`, builds the service image and imports it, applies `k8s/app/parser.yaml`, and
waits until `/parse` answers through the Ingress.

```bash
curl -i -XPOST http://localhost:8080/parse -H 'Content-Type: application/json' \
     -d '{"text": "set an alarm for seven thirty am tomorrow"}'
# HTTP/1.1 200 OK   X-Model-Version: v1   X-Served-By: parser-...
# {"intent":"alarm_set","date":"tomorrow","time":"seven thirty am","place":null}
```

## Test

```bash
make lint            # ruff
make test            # 18 unit tests (sharding, cleaning rules, schema validation, API, FAULT_INJECT)
make nacl            # nacl_sim.py on network/nacl.json + the sample packets
make verify-clean    # re-run clean.py on the committed shards and compare with datasets/v1/train.sha256
make train SUBSET=50 # quick training check (writes models/subset-50/, training/loss_subset50.csv)
make loadtest        # the standard Locust command against http://localhost:8080
```

The Task 4/7 experiments are scripted: `scripts/experiments.sh {coldstart|recovery|rollout|hpa|lb|standard}`;
the Task 3 limits test is `scripts/task3_container_test.sh`.

## Shut down

```bash
make down            # k3d cluster delete cc-kd (removes SeaweedFS data too)
```

## Other targets

| Target | What it does |
|---|---|
| `make label OLLAMA_URL=http://host.docker.internal:11434` | Build the worker image and run the 2-shard Indexed Job in namespace `pipeline` (Ollama + phi3.5 on the host) |
| `make clean-data` | `worker/clean.py`: shards in S3 → `datasets/v1/train.jsonl` + `train.sha256` |
| `make train` / `make train-v2` | Train v1 (seed 42) / v2 (seed 7) from `training/config.yaml` |
| `make eval-student` | Predictions + `score.py` report for v1 on validation |
| `make images`, `make deploy` | Rebuild/import the service image, re-apply manifests |

## Repository layout

```
cluster/k3d.yaml            k3d cluster definition
k8s/data/seaweedfs.yaml     SeaweedFS (S3) in namespace data
k8s/pipeline/teacher-job.yaml  2-way Indexed Job for teacher labelling
k8s/app/parser.yaml         Deployment, Service, Ingress, HPA in namespace app
prompts/teacher_prompt.txt  teacher prompt
worker/                     teacher client, shard worker, clean.py, Dockerfile
datasets/teacher/           raw shard outputs (teacher replies exactly as generated)
datasets/v1/                cleaned train.jsonl + train.sha256
training/                   config.yaml, train.py, predict.py, loss.csv
models/v1, models/v2        model.sha256 (weights live only in SeaweedFS and the GitHub Release)
service/                    FastAPI app, inference + schema validation, Dockerfile, pinned requirements
network/                    subnets.md, nacl.json, packets_sample.jsonl
tests/                      unit tests
.github/workflows/ci.yml    lint → unit tests → model evaluation (Release asset) → quality gate; gitleaks
results/                    logs, score reports, Locust outputs, command outputs
data/, tools/               the provided student kit (unchanged)
```

Data: commands from SLURP (Bastianelli et al., EMNLP 2020), CC BY 4.0. Student model: EleutherAI/pythia-70m, Apache 2.0.
