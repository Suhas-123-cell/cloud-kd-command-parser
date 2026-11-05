"""Command-parser inference service.

POST /parse   {"text": "..."} -> 200 + the four-key JSON object (headers X-Model-Version, X-Served-By)
                                -> 500 {"error": "invalid_model_output"} if the output fails schema.json
                                -> 503 while the model is still loading
GET  /healthz -> 200 while the process is alive
GET  /readyz  -> 503 until the model is loaded, then 200

Configuration (environment):
  MODEL_URI        s3://models/v1/          prefix holding the model + tokenizer files
  MODEL_VERSION    v1                       returned in X-Model-Version
  S3_ENDPOINT      http://seaweedfs.data.svc.cluster.local:8333
  MAX_NEW_TOKENS   64                       generation cap
  FAULT_INJECT     0|1                      1 replaces every model output with "not json"
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from the Kubernetes Secret s3-credentials
  POD_NAME         set from the downward API; falls back to the hostname
"""
import logging
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import boto3
from botocore.config import Config
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference import Parser, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("parser")

MODEL_URI = os.environ.get("MODEL_URI", "s3://models/v1/")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0") == "1"
SERVED_BY = os.environ.get("POD_NAME") or socket.gethostname()
THREADS = int(os.environ.get("TORCH_THREADS", "1"))

state = {"parser": None, "error": None}
lock = threading.Lock()  # one generation at a time per replica (CPU-bound; avoids thrashing)
app = FastAPI(title="command-parser")


def download_model(uri, dest):
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, region_name="us-east-1",
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    n = 0
    for o in objs:
        rel = o["Key"][len(prefix):].lstrip("/")
        if not rel or "/" in rel or rel.endswith((".tar.gz", ".sha256")):
            continue  # only the flat model + tokenizer files
        s3.download_file(bucket, o["Key"], str(Path(dest) / rel))
        n += 1
        log.info("downloaded s3://%s/%s (%d bytes)", bucket, o["Key"], o["Size"])
    if n == 0:
        raise RuntimeError(f"no model files under {uri}")


def load_model():
    t0 = time.time()
    try:
        dest = tempfile.mkdtemp(prefix="model-")
        log.info("model download start: %s from %s", MODEL_URI, S3_ENDPOINT)
        download_model(MODEL_URI, dest)
        log.info("model download done in %.1fs", time.time() - t0)
        state["parser"] = Parser(dest, max_new_tokens=MAX_NEW_TOKENS, threads=THREADS)
        shutil.rmtree(dest, ignore_errors=True)
        log.info("model %s ready in %.1fs (pod %s, fault_inject=%s)",
                 MODEL_VERSION, time.time() - t0, SERVED_BY, FAULT_INJECT)
    except Exception as e:  # stay unready; the liveness probe keeps passing, logs show why
        state["error"] = repr(e)
        log.exception("model load failed")


@app.on_event("startup")
def startup():
    threading.Thread(target=load_model, daemon=True).start()


class ParseRequest(BaseModel):
    text: str


def headers():
    return {"X-Model-Version": MODEL_VERSION, "X-Served-By": SERVED_BY}


@app.post("/parse")
def parse(req: ParseRequest):
    parser = state["parser"]
    if parser is None:
        return JSONResponse({"error": "model_not_ready"}, status_code=503, headers=headers())
    with lock:
        raw = parser.generate(req.text)
    if FAULT_INJECT:
        raw = "not json"
    obj = validate(raw)
    if obj is None:
        log.warning("invalid model output for %r: %r", req.text, raw)
        return JSONResponse({"error": "invalid_model_output"}, status_code=500, headers=headers())
    return JSONResponse(obj, status_code=200, headers=headers())


@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if state["parser"] is None:
        return JSONResponse({"status": "loading", "error": state["error"]}, status_code=503)
    return {"status": "ready", "model_version": MODEL_VERSION}
