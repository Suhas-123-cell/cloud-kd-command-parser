# CS3081 mini project: distil phi3.5 -> pythia-70m and serve it on k3d.
#   make up      build and deploy everything (cluster, SeaweedFS, model from the GitHub Release, service)
#   make down    delete the cluster
#   make train   train the student (make train SUBSET=50 for a quick check)
SHELL := /bin/bash
.DEFAULT_GOAL := help

CLUSTER     := cc-kd
REPO        ?= Suhas-123-cell/cloud-kd-command-parser
RELEASE_TAG ?= models
PY          := .venv/bin/python
TORCH_INDEX := https://download.pytorch.org/whl/cpu
OLLAMA_URL  ?= http://host.docker.internal:11434
PARSER_IMG  := cc-kd/parser:1.0
WORKER_IMG  := cc-kd/worker:1.0
SUBSET      ?=
HOST        ?= http://localhost:8080

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

# ---------- Python environments ----------
.venv/.tools: requirements-dev.txt
	@[ -x $(PY) ] || { if command -v uv >/dev/null; then uv venv -q -p 3.12 .venv; else python3 -m venv .venv; fi; }
	@if command -v uv >/dev/null; then uv pip install -q -p $(PY) -r requirements-dev.txt; \
	 else $(PY) -m pip install -q -r requirements-dev.txt; fi
	@touch $@

.venv/.train: .venv/.tools training/requirements.txt
	@if command -v uv >/dev/null; then \
	   uv pip install -q -p $(PY) --index-url $(TORCH_INDEX) torch==2.5.1 && \
	   uv pip install -q -p $(PY) -r training/requirements.txt; \
	 else $(PY) -m pip install -q --index-url $(TORCH_INDEX) torch==2.5.1 && \
	   $(PY) -m pip install -q -r training/requirements.txt; fi
	@touch $@

venv: .venv/.tools ## create the tools virtualenv

# ---------- Whole system ----------
up: .venv/.tools cluster storage seed-data seed-models images deploy ## build + deploy everything from a fresh clone
	@echo "Ready: curl -s -XPOST $(HOST)/parse -H 'Content-Type: application/json' -d '{\"text\":\"set an alarm for seven am\"}'"

down: ## delete the cluster (and everything in it)
	k3d cluster delete $(CLUSTER)

cluster: ## create the k3d cluster from cluster/k3d.yaml (no-op if it exists)
	@k3d cluster list $(CLUSTER) >/dev/null 2>&1 || k3d cluster create --config cluster/k3d.yaml
	kubectl config use-context k3d-$(CLUSTER)

storage: ## deploy SeaweedFS in namespace data and create the Secrets
	kubectl apply -f k8s/data/seaweedfs.yaml
	./scripts/secrets.sh
	kubectl -n data rollout status deploy/seaweedfs --timeout=300s
	@for i in $$(seq 1 30); do $(PY) scripts/s3.py mb datasets models >/dev/null 2>&1 && break; sleep 2; done

seed-data: .venv/.tools ## buckets + commands_pool + teacher shards + cleaned dataset
	$(PY) scripts/s3.py mb datasets models
	$(PY) scripts/s3.py put data/commands_pool.jsonl s3://datasets/commands_pool.jsonl
	$(PY) scripts/s3.py putdir datasets/teacher s3://datasets/teacher/
	$(PY) scripts/s3.py putdir datasets/v1 s3://datasets/v1/

seed-models: .venv/.tools ## download model-v1/v2 from the GitHub Release, verify checksums, upload to s3://models
	./scripts/seed_models.sh $(REPO) $(RELEASE_TAG)

images: ## build the service image and import it into the cluster
	docker build -f service/Dockerfile -t $(PARSER_IMG) .
	k3d image import -c $(CLUSTER) $(PARSER_IMG)

deploy: ## apply the app manifests and wait for the rollout
	kubectl apply -f k8s/app/parser.yaml
	kubectl -n app rollout status deploy/parser --timeout=600s

# ---------- Task 1: teacher labelling ----------
label: ## run the two-shard teacher Job in namespace pipeline (needs Ollama + phi3.5 on the host)
	docker build -f worker/Dockerfile -t $(WORKER_IMG) .
	k3d image import -c $(CLUSTER) $(WORKER_IMG)
	kubectl -n pipeline delete job teacher-label --ignore-not-found
	sed 's#http://host.docker.internal:11434#$(OLLAMA_URL)#' k8s/pipeline/teacher-job.yaml | kubectl apply -f -
	kubectl -n pipeline wait --for=condition=complete job/teacher-label --timeout=6h

clean-data: .venv/.tools ## clean the shard outputs -> datasets/v1/train.jsonl + train.sha256 (and upload)
	$(PY) worker/clean.py

verify-clean: .venv/.tools ## re-run clean.py on the committed shards and compare with train.sha256
	$(PY) worker/clean.py --local-dir datasets/teacher --out-dir /tmp/cc-kd-verify --counts /tmp/cc-kd-verify/counts.json
	@diff <(cut -d' ' -f1 /tmp/cc-kd-verify/train.sha256) <(cut -d' ' -f1 datasets/v1/train.sha256) && echo "checksum matches datasets/v1/train.sha256"

# ---------- Task 2: training + evaluation ----------
train: .venv/.train ## train the student; make train SUBSET=50 for the quick check
	$(PY) training/train.py --config training/config.yaml $(if $(SUBSET),--subset $(SUBSET) --output-dir models/subset-$(SUBSET) --loss-csv training/loss_subset$(SUBSET).csv)

eval-student: .venv/.train ## score models/v1 on validation.jsonl
	$(PY) training/predict.py models/v1/model-v1 data/validation.jsonl results/student_v1_validation_predictions.jsonl
	python3 tools/score.py results/student_v1_validation_predictions.jsonl data/validation.jsonl --out results/student_v1_validation.json

# ---------- Tests ----------
lint: .venv/.tools ## ruff
	$(PY) -m ruff check .

test: .venv/.tools ## unit tests
	$(PY) -m pytest -q tests

nacl: ## check the NACL against the sample packets
	python3 tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check

loadtest: .venv/.tools ## the standard 2-minute Locust run against the Ingress
	.venv/bin/locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host $(HOST) --csv results/loadtest

.PHONY: help venv up down cluster storage seed-data seed-models images deploy label clean-data verify-clean train eval-student lint test nacl loadtest
