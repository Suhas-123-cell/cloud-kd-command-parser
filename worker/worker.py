"""Teacher-labelling worker.

Reads the command pool from object storage, keeps the commands whose 0-based line number
satisfies  line % SHARD_COUNT == SHARD_INDEX, asks the teacher for each one, and writes
{"id", "text", "teacher_raw"} lines to its own object  s3://datasets/teacher/shard-<i>.jsonl.

The shard object is re-uploaded every CHECKPOINT_EVERY records, and a restarted worker
resumes from it, so a killed Pod loses at most one checkpoint of work.

Environment
  SHARD_INDEX, SHARD_COUNT        required
  S3_ENDPOINT                     e.g. http://seaweedfs.data.svc.cluster.local:8333
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   from the Kubernetes Secret s3-credentials
  OLLAMA_URL                      e.g. http://host.docker.internal:11434
  INPUT_URI  (default s3://datasets/commands_pool.jsonl)
  OUTPUT_PREFIX (default s3://datasets/teacher/)
"""
import json
import os
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from teacher import ask_teacher, load_prompt


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def split_uri(uri):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def select_shard(lines, shard_index, shard_count):
    """Return the records whose 0-based line number % shard_count == shard_index."""
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"SHARD_INDEX {shard_index} must be in [0, {shard_count})")
    out = []
    for n, line in enumerate(lines):
        if n % shard_count == shard_index:
            out.append(json.loads(line))
    return out


def read_object(s3, bucket, key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def main():
    shard_index = int(os.environ["SHARD_INDEX"])
    shard_count = int(os.environ["SHARD_COUNT"])
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", "50"))
    in_bucket, in_key = split_uri(os.environ.get("INPUT_URI", "s3://datasets/commands_pool.jsonl"))
    out_bucket, out_prefix = split_uri(os.environ.get("OUTPUT_PREFIX", "s3://datasets/teacher/"))
    out_key = f"{out_prefix.rstrip('/')}/shard-{shard_index}.jsonl"

    s3 = s3_client()
    pool = read_object(s3, in_bucket, in_key)
    lines = [ln for ln in pool.splitlines() if ln.strip()]
    mine = select_shard(lines, shard_index, shard_count)

    existing = read_object(s3, out_bucket, out_key) or ""
    done = [json.loads(ln) for ln in existing.splitlines() if ln.strip()]
    done_ids = {r["id"] for r in done}
    print(f"shard {shard_index}/{shard_count}: {len(mine)} commands, {len(done)} already done, "
          f"-> s3://{out_bucket}/{out_key}", flush=True)

    template = load_prompt()
    t0 = time.time()
    new = 0
    for rec in mine:
        if rec["id"] in done_ids:
            continue
        for attempt in range(5):
            try:
                raw = ask_teacher(template, rec["text"])
                break
            except Exception as e:  # network blip or Ollama restart: retry, do not skip
                print(f"{rec['id']}: attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            sys.exit(f"giving up on {rec['id']}")
        done.append({"id": rec["id"], "text": rec["text"], "teacher_raw": raw})
        new += 1
        if new % checkpoint_every == 0:
            s3.put_object(Bucket=out_bucket, Key=out_key, Body=to_jsonl(done))
            rate = new / (time.time() - t0)
            print(f"shard {shard_index}: {len(done)}/{len(mine)} ({rate:.2f} cmd/s)", flush=True)

    s3.put_object(Bucket=out_bucket, Key=out_key, Body=to_jsonl(done))
    print(f"shard {shard_index}: finished {len(done)}/{len(mine)} records "
          f"({new} new) in {time.time() - t0:.0f}s", flush=True)


def to_jsonl(records):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


if __name__ == "__main__":
    main()
