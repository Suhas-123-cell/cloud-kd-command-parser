"""Clean the teacher shard outputs into datasets/v1/train.jsonl (+ train.sha256).

python worker/clean.py                       # read shards from s3://datasets/teacher/, upload results
python worker/clean.py --local-dir DIR       # read DIR/shard-*.jsonl instead (no upload unless --upload)

Each record is checked in this order and rejected for the FIRST rule it breaks:
  (a) teacher_raw is not valid JSON
  (b) the JSON is not an object whose keys are exactly intent, date, time, place
  (c) intent is not one of the eight intents in data/schema.json
  (d) a slot (date/time/place) is neither null nor a string that appears in the command
      (case-insensitive, whitespace-normalised substring match)
Records that pass (a)-(d) are then de-duplicated on the command text (lower-cased,
whitespace-normalised); for each duplicated command the record with the smallest id is kept.
Kept records are written in id order, exactly as stored in the shards (teacher_raw untouched).
So: input = a + b + c + d + duplicates + kept.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "data" / "schema.json").read_text())
KEYS = set(SCHEMA["keys"])
INTENTS = set(SCHEMA["intents"])
SLOTS = ("date", "time", "place")


def norm(s):
    return " ".join(s.lower().split())


def id_key(rid):
    m = re.search(r"\d+", rid)
    return (int(m.group()) if m else -1, rid)


def classify(rec):
    """Return None if the record is good, else the rejection reason 'a'..'d'."""
    try:
        obj = json.loads(rec["teacher_raw"])
    except (json.JSONDecodeError, TypeError):
        return "a"
    if not isinstance(obj, dict) or set(obj.keys()) != KEYS:
        return "b"
    if obj["intent"] not in INTENTS:
        return "c"
    text = norm(rec["text"])
    for k in SLOTS:
        v = obj[k]
        if v is None:
            continue
        if not isinstance(v, str) or not norm(v) or norm(v) not in text:
            return "d"
    return None


def clean(records):
    counts = {"input": len(records), "a_invalid_json": 0, "b_wrong_keys": 0,
              "c_bad_intent": 0, "d_bad_slot": 0, "duplicates": 0, "kept": 0}
    names = {"a": "a_invalid_json", "b": "b_wrong_keys", "c": "c_bad_intent", "d": "d_bad_slot"}
    good = []
    for rec in sorted(records, key=lambda r: id_key(r["id"])):
        reason = classify(rec)
        if reason:
            counts[names[reason]] += 1
        else:
            good.append(rec)
    seen, kept = set(), []
    for rec in good:  # already in id order, so the first occurrence has the smallest id
        t = norm(rec["text"])
        if t in seen:
            counts["duplicates"] += 1
            continue
        seen.add(t)
        kept.append(rec)
    counts["kept"] = len(kept)
    assert counts["input"] == sum(v for k, v in counts.items() if k != "input")
    return kept, counts


def to_jsonl(records):
    return "".join(
        json.dumps({"id": r["id"], "text": r["text"], "teacher_raw": r["teacher_raw"]},
                   ensure_ascii=False) + "\n"
        for r in records
    ).encode("utf-8")


def read_shards_local(d):
    recs = []
    for p in sorted(Path(d).glob("shard-*.jsonl")):
        recs += [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return recs


def s3_client():
    sys.path.insert(0, str(ROOT / "scripts"))
    from s3 import client
    return client()


def read_shards_s3(s3, bucket, prefix):
    recs = []
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
                  if re.search(r"shard-\d+\.jsonl$", o["Key"]))
    for k in keys:
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read().decode("utf-8")
        recs += [json.loads(ln) for ln in body.splitlines() if ln.strip()]
        print(f"read s3://{bucket}/{k}")
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", help="read shard-*.jsonl from this directory instead of S3")
    ap.add_argument("--out-dir", default=str(ROOT / "datasets" / "v1"))
    ap.add_argument("--counts", default=str(ROOT / "results" / "clean_counts.json"))
    ap.add_argument("--upload", action="store_true", help="upload results (default when reading S3)")
    a = ap.parse_args()

    s3 = None
    if a.local_dir:
        records = read_shards_local(a.local_dir)
    else:
        s3 = s3_client()
        records = read_shards_s3(s3, "datasets", "teacher/")

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        sys.exit("an id appears in more than one shard record")

    kept, counts = clean(records)
    data = to_jsonl(kept)
    digest = hashlib.sha256(data).hexdigest()

    os.makedirs(a.out_dir, exist_ok=True)
    Path(a.out_dir, "train.jsonl").write_bytes(data)
    Path(a.out_dir, "train.sha256").write_text(f"{digest}  train.jsonl\n")
    Path(a.counts).parent.mkdir(parents=True, exist_ok=True)
    Path(a.counts).write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"sha256 {digest}")

    if a.upload or s3 is not None:
        s3 = s3 or s3_client()
        s3.put_object(Bucket="datasets", Key="v1/train.jsonl", Body=data)
        s3.put_object(Bucket="datasets", Key="v1/train.sha256", Body=f"{digest}  train.jsonl\n".encode())
        print("uploaded s3://datasets/v1/train.jsonl and train.sha256")


if __name__ == "__main__":
    main()
